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
create_folder('./trained_models')

# gen_shp_labels(map_list, obj_name, input_dir, output_dir)

tif_dir, png_dir = input_dir, output_dir
num_pos, num_neg = args.num_pos_samples, args.num_neg_samples

# all_x_img, all_node_cat_targets, all_node_reg_targets, all_nodes_inputs, all_adj_targets, all_conn_targets, all_nodes_mask = \
#     gen_labels4train(map_list, tif_dir, png_dir, obj_name, num_pos=num_pos, num_neg=num_neg)

gen_labels4train(map_list, tif_dir, png_dir, obj_name, num_pos=num_pos, num_neg=num_neg)
