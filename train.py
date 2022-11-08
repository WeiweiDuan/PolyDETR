from models.graph import build
import argparse
import sys, os
import torch
from util.misc import nested_tensor_from_tensor_list, NestedTensor
import numpy as np
import cv2
import copy
from data_process.gen_targets import gen_shp_labels, gen_labels4train
from util.args import get_args_parser
from util.util_helper import remove_files, create_folder
import glob

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(args.device)

model, criterion = build(args)
model.to(device)

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
model_name = args.saved_model_name

map_name_str = args.map_name_list
map_list = map_name_str.split(',')

obj_name = args.obj_name
input_dir = args.input_data_dir
output_dir = args.output_data_dir

# remove_files(output_dir)
# create_folder('./trained_models')

# gen_shp_labels(map_list, obj_name, input_dir, output_dir)

tif_dir, png_dir = input_dir, output_dir
num_pos, num_neg = args.num_pos_samples, args.num_neg_samples

# all_x_img, all_node_cat_targets, all_node_reg_targets, all_nodes_inputs, all_adj_targets, all_conn_targets, all_nodes_mask = \
#     gen_labels4train(map_list, tif_dir, png_dir, obj_name, num_pos=num_pos, num_neg=num_neg)

sample_dir = os.path.join(output_dir, f'{obj_name}{args.samples_folder_suffix}')
print(sample_dir)

def retrieve_samples(sample_paths):
    all_x_img, all_node_cat_targets, all_node_reg_targets, all_nodes_inputs, all_adj_targets, all_conn_targets, all_nodes_mask\
        = [], [], [], [], [], [], []
    for sample_path in sample_paths:
        data = np.load(sample_path, allow_pickle=True)
        all_x_img.append(data['x_img'])
        all_node_cat_targets.append(data['node_cat_targets'])
        all_node_reg_targets.append(data['node_reg_targets'])
        all_nodes_inputs.append(data['node_inputs'])
        all_adj_targets.append(data['adj_targets'])
        all_conn_targets.append(data['conn_targets'])
        all_nodes_mask.append(data['nodes_mask'])
    all_x_img = np.stack(all_x_img)
    all_node_cat_targets = np.stack(all_node_cat_targets)
    all_node_reg_targets = np.stack(all_node_reg_targets)
    all_nodes_inputs = np.stack(all_nodes_inputs)
    all_adj_targets = np.stack(all_adj_targets)
    all_conn_targets = np.stack(all_conn_targets)
    all_nodes_mask = np.stack(all_nodes_mask)
        
    all_node_reg_targets = all_node_reg_targets/args.grid_size
    all_nodes_inputs = all_nodes_inputs/(args.img_size-1.0)

    x_train_color = np.array(all_x_img).astype(np.float64)
    x_train = x_train_color/255.0
    x_train = (x_train - img_mean) / img_std
    
    x_torch = torch.from_numpy(x_train.copy()).type(torch.FloatTensor).to(device)
    x_torch = torch.moveaxis(x_torch, 3, 1)
    
    targets = {}

    enc_cat_targets_ts = torch.from_numpy(all_node_cat_targets.copy()).type(torch.LongTensor).to(device)
    enc_reg_targets_ts = torch.from_numpy(all_node_reg_targets.copy()).type(torch.FloatTensor).to(device)
    dec_inputs_ts = torch.from_numpy(all_nodes_inputs.copy()).type(torch.FloatTensor).to(device)
    dec_adj_targets_ts = torch.from_numpy(all_adj_targets.copy()).type(torch.FloatTensor).to(device)
    dec_conn_targets_ts = torch.from_numpy(all_conn_targets.copy()).type(torch.FloatTensor).to(device)
    dec_mask_ts = torch.from_numpy(all_nodes_mask.copy()).type(torch.BoolTensor).to(device)

    targets['cat_nodes'] = enc_cat_targets_ts
    targets['reg_nodes'] = enc_reg_targets_ts
    targets['line'] = dec_adj_targets_ts
    targets['conn'] = dec_conn_targets_ts
    targets['mask'] = dec_mask_ts
    
    enc_inputs_nested = NestedTensor(x_torch, x_torch)
    dec_inputs_nested = NestedTensor(dec_inputs_ts, dec_mask_ts)
    return targets, enc_inputs_nested, dec_inputs_nested

model_without_ddp = model

###################################
##### load trained model ##########
###################################
if args.resume and args.trained_model_path != None:
    checkpoint = torch.load('%s' %(args.trained_model_path))
    model.load_state_dict(checkpoint['model'])
    torch.cuda.empty_cache()
    print('loading trained model from {}'.format(args.trained_model_path))

param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]

# optimizer = torch.optim.Adadelta(param_dicts, lr=args.lr)
optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

max_norm = args.clip_max_norm
s_epoch = args.start_epoch
epoch = args.epochs
bs = args.batch_size
prev_loss = 10000.0
# num_img = x_torch.shape[0]-1
# indices = np.arange(0, num_img, 1, dtype=int)

all_sample_paths = glob.glob(os.path.join(sample_dir, '*.npz'))
print('Total samples', len(all_sample_paths))

for e in range(s_epoch, s_epoch+epoch):
    print(f'Epoch {e} starts ...')
    model.train()
    criterion.train()
    total_loss, enc_cat_loss, enc_reg_loss, dec_adj_loss, aux_loss = 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(0, len(all_sample_paths), bs):
        sample_paths = np.random.choice(all_sample_paths, size=bs, replace=False)   
        sub_target, enc_inputs_nested, dec_inputs_nested = retrieve_samples(sample_paths)

        outputs, enc_attn, dec_attn = model( enc_inputs_nested, dec_inputs_nested)

        loss_dict = criterion(outputs, sub_target)

        losses = sum(loss_dict[k] for k in loss_dict.keys())
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        losses.detach_()
        torch.cuda.empty_cache()
#         total_loss += losses.item()
        enc_cat_loss += loss_dict['loss_cat_nodes'].item()
        enc_reg_loss += loss_dict['loss_reg_nodes'].item()
        dec_adj_loss += loss_dict['loss_adj'].item()
        aux_loss += loss_dict['loss_conn_0'].item()+loss_dict['loss_conn_2'].item()+loss_dict['loss_conn_3'].item()+\
            loss_dict['loss_conn_4'].item()+loss_dict['loss_conn_5'].item()
        total_loss += enc_cat_loss + enc_reg_loss + dec_adj_loss
    print('%d epoch, lr=%f, total loss=%f, cat loss=%f, reg loss=%f, line loss=%f, aux loss=%f'%\
              (e, optimizer.param_groups[1]["lr"], total_loss, enc_cat_loss, enc_reg_loss, dec_adj_loss, aux_loss))
    lr_scheduler.step()
    
    if total_loss < prev_loss:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
                }, 'trained_models/%s'%(model_name))
        prev_loss = total_loss 
    if e % 30 == 0 and e > 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
                }, 'trained_models/%s'%(model_name[:-4]+'_e%d.pth'%(e)))

torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
                }, 'trained_models/%s'%(model_name[:-4]+'_final.pth'))