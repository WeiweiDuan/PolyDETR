import argparse
import sys, os
import torch
import numpy as np
import cv2
import copy
from data_process.gen_targets import gen_shp_labels, gen_labels4train
from util.args import get_args_parser
from util.util_helper import remove_files, create_folder

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

map_name_str = args.map_name_list
map_list = map_name_str.split(',')

obj_name = args.obj_name
input_dir = args.input_data_dir
output_dir = args.output_data_dir

# remove_files(output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
gen_shp_labels(map_list, obj_name, input_dir, output_dir)