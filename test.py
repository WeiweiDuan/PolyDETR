from models.graph import build
import argparse
import sys, os
import torch
import numpy as np
import cv2
import copy
from util.args import get_args_parser
from util.tif2png2shp import tif2png
import networkx as nx
from test_helper import construct_graph_prob
from util.args_test import get_args_parser
from util.util_helper import remove_files, create_folder
import time
from util.buf4competition import buff_main

start = time.time()
parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(args.device)

model_path = args.trained_model_path
output_dir = args.pred_dir
DATA_DIR = args.png_map_dir
MAP_PATH = os.path.join(DATA_DIR, args.map_name+'.png')
pred_name = args.map_name+'_'+args.obj_name+'_pred'

create_folder(output_dir)

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])
args.dropout = 0.0
device = torch.device(args.device)

model, criterion = build(args)
model.to(device)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
torch.cuda.empty_cache()

batch = args.batch_size
win_size = args.img_size
buffer = args.buffer_size
thres = args.adj_prob_thres
stride = args.crop_stride
vote_threhold = args.vote_thres
gamma = 1.0
num_nodes_thres = 10
dist_thres = 40

if not os.path.exists(MAP_PATH):
    tif_map_path = os.path.join(args.tif_map_dir, args.map_name+'.tif')
    tif2png(tif_map_path, MAP_PATH)

map_img = cv2.imread(MAP_PATH)

height, width = map_img.shape[:2]
pred_graph = np.zeros(map_img.shape[:2])

# for i in range(0, height, stride):
#     for j in range(0, width, stride):
#         x_test_name.append(str(i)+'_'+str(j))
        

# nodes_dict, edges_dict = construct_graph_prob(model, x_test_name, map_img, device,\
#                                               win_size=win_size, batch=batch, thres=thres, buffer=buffer,\
#                                               num_nodes_thres=num_nodes_thres, dist_thres=dist_thres)

sub_height, sub_width = 1000, 1000

for height_start in range(0, height, sub_height):
    for width_start in range(0, width, sub_width):
        height_end = min(height_start+sub_height, height)
        width_end = min(width_start+sub_width, width)
        x_test_name = []
        for i in range(height_start, height_end, stride):
            for j in range(width_start, width_end, stride):
                x_test_name.append(str(i)+'_'+str(j))
        
        nodes_dict, edges_dict = construct_graph_prob(model, x_test_name, map_img, device,\
                                                      win_size=win_size, batch=batch, thres=thres, buffer=buffer,\
                                                      num_nodes_thres=num_nodes_thres, dist_thres=dist_thres)

        print('==== drawing lines for subregion [%d:%d, %d:%d] ===='%(height_start, height_end, width_start, width_end))
        ##### draw the predicted polylines on the pred image
        for edge, count in edges_dict.items():
        ##### connect #votes > 1
            if count < vote_threhold:
                continue
            x1, y1 = nodes_dict[edge[0]]
            x2, y2 = nodes_dict[edge[1]]
            cv2.line(pred_graph, (y1,x1), (y2,x2), 255, 1)

cv2.imwrite(os.path.join(output_dir, pred_name+'_graph.png'), pred_graph)
print('save the predicted map in %s'%(os.path.join(output_dir, pred_name+'_graph.png')))

##### dilate the drawn polylines to conflate very close polylines into one
kernel = np.ones((3,3), np.uint8)
pred_dilate = cv2.dilate(pred_graph, kernel, iterations=2)

##### thining the results into 1-pixel width
pred_thin = np.zeros_like(pred_dilate)
pred_thin = cv2.ximgproc.thinning(pred_dilate.astype('uint8'), thinningType=cv2.ximgproc.THINNING_GUOHALL)

# cv2.imwrite(os.path.join(output_dir, pred_name+'.png'), pred_thin)
# print('save the predicted map in %s'%(os.path.join(output_dir, pred_name+'.png')))

##### buffer the results for the competition
buf_pred = buff_main(pred_thin)
cv2.imwrite(os.path.join(output_dir, pred_name+'.png'), buf_pred)
print('saved the buffered prediction in %s'%(os.path.join(output_dir, pred_name+'.png')))

end = time.time()
print('predicting {} takes {} minutes.'.format(pred_name, (end-start)//60))
