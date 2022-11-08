import os, cv2, math, sys
import numpy as np
sys.path.append(".")
from .data_loader import data_generator
from .gen_refine_targets_helper import construct_sub_graph, gen_node_cls_reg_targets, gen_adj_conn_targets
import networkx as nx
import copy
from util.util_helper import remove_files
from util.rotation_operator import rotate_nodes_dict, rotate
from util.process_shp import read_shp, interpolation, construct_graph_on_map, interpolate_polylines
from util.args import get_args_parser
import glob
import random 
import torchvision
import PIL

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])

def color_aug(img_array):
    color_aug = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3)    
    img_p =  PIL.Image.fromarray(np.uint8(img_array))
    aug_img_p = color_aug(img_p) 
    aug_img = np.array(aug_img_p)
    return aug_img

def gen_nodes_edges_targets(data_dir, tif_dir, png_name, raster_label_name, map_obj_name, obj_name, num_pos, num_neg):
    #############################################
    ############### parameters ###############
    #############################################
    png_path = os.path.join(data_dir, png_name)
    raster_label_path = os.path.join(data_dir, raster_label_name) 
    OBJECT_LIST = [map_obj_name]

    OBJECT_NUMS = [num_pos, num_neg] 
    WIN_SIZE = args.img_size
    num_nodes = args.num_nodes
    NB_CLASSES = 2
    gamma = 1.0
    shift_augment = True
    if WIN_SIZE == 512:
        shift_range = (160, 160)
    else:
        shift_range = (80, 80)
    num_shift = 1
    times4multi = 1
    inter_dis = 16

    tif_name = png_name[:-4]+'.tif'
    shp_name = raster_label_name[:-4]+'.shp'

    shp_path = os.path.join(data_dir, shp_name)
    tif_path = os.path.join(tif_dir, tif_name)
    #############################################
    #############################################
    
    #############################################
    ########## process shp input ################
    #############################################
    polylines = read_shp(shp_path, tif_path)
    polylines_interp = interpolate_polylines(polylines, inter_dis=inter_dis)
    map_nodes_dict, map_edges_list = construct_graph_on_map(polylines_interp)
    #############################################
    #############################################
    
    #############################################
    ########## get x_inputs #####################
    #############################################
    all_x_train, all_y_train, all_img_indices = \
        data_generator(data_dir, png_path, raster_label_path, OBJECT_LIST, OBJECT_NUMS, WIN_SIZE, NB_CLASSES, \
                       shift_augment=shift_augment, shift_range=shift_range, num_shift=num_shift, times4multi=times4multi,\
                       gamma=gamma, random=True)
    
    x_train, y_train, img_indices = [], [], []
    for i, x_img in enumerate(all_x_train):
        x_train.append(all_x_train[i])
        y_train.append(all_y_train[i])
        img_indices.append(all_img_indices[i])
        
    
    all_x_img, all_node_cat_targets, all_node_reg_targets, all_nodes_inputs, all_adj_targets, all_conn_targets, all_nodes_mask\
        = [], [], [], [], [], [], []
    
    for i, x_img in enumerate(x_train):
        xmin, ymin, xmax, ymax = img_indices[i][0]-WIN_SIZE//2, img_indices[i][1]-WIN_SIZE//2, \
                                img_indices[i][0]+WIN_SIZE//2, img_indices[i][1]+WIN_SIZE//2
        sub_nodes_dict, sub_edges_list = construct_sub_graph(map_nodes_dict, map_edges_list, [xmin, ymin, xmax, ymax])

        for r_angle in random.sample([90, 180, 270, 360], 1):
            rot_sub_nodes_dict = rotate_nodes_dict(sub_nodes_dict, r_angle, cent=[WIN_SIZE//2,WIN_SIZE//2])
            node_cat_targets, node_reg_targets, nodes_in_grids, rot_sub_nodes_dict, sub_edges_list \
                    = gen_node_cls_reg_targets(rot_sub_nodes_dict, sub_edges_list, gsize=args.grid_size, img_size=WIN_SIZE)

            node_inputs, adj_targets, conn_targets, nodes_mask = \
                gen_adj_conn_targets(nodes_in_grids, rot_sub_nodes_dict, sub_edges_list, gsize=args.grid_size, img_size=WIN_SIZE, n_nodes=num_nodes)
            
            rot_x_img = rotate(x_img, angle=r_angle)
            # color augmentation
            new_img = color_aug(rot_x_img)
            # write sample to disk
            sample_idx = len(glob.glob(os.path.join(data_dir, f'{obj_name}{args.samples_folder_suffix}', '*.npz')))
            file_name = os.path.join(data_dir, f'{obj_name}{args.samples_folder_suffix}', f'{sample_idx}.npz')
            data = {
                'x_img': new_img,
                'node_cat_targets': node_cat_targets,
                'node_reg_targets': node_reg_targets,
                'node_inputs': node_inputs,
                'adj_targets': adj_targets,
                'conn_targets': conn_targets,
                'nodes_mask': nodes_mask}
            np.savez(file_name, **data)
        