import os
import numpy as np
from data_process.gen_targets4single_map import gen_nodes_edges_targets
import sys
from vectorization.png2shp import build_graph
from util.tif2png2shp import tif2png, gen_txt_labels
from util.args import get_args_parser

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])

def gen_shp_labels(map_list, obj_name, input_dir, output_dir):
    for map_name in map_list:
        ##### tif2png ######
        tif_map_path = os.path.join(input_dir, map_name+'.tif')
        png_map_path = os.path.join(output_dir, map_name+'.png')
        
        tif_label_path = os.path.join(input_dir, map_name+'_'+obj_name+'.tif')
        png_label_path = os.path.join(output_dir, map_name+'_'+obj_name+'.png')
        
#         if os.path.exists(png_map_path) and  os.path.exists(png_label_path):
#             continue
        
        tif2png(tif_map_path, png_map_path)
        tif2png(tif_label_path, png_label_path)
        
        ##### tif2shp #####
        png_label_name = map_name+'_'+obj_name+'.png'
        shp_label_name = map_name+'_'+obj_name+'.shp'
        shp_label_path = os.path.join(output_dir, shp_label_name)
        epsg = 4269 
        coor_sys_path = tif_map_path
        raster_path = png_label_path      
        build_graph(raster_path, coor_sys_path, epsg, shp_label_path)
        
        ##### shp2txt #####
        txt_output_path = os.path.join(output_dir, map_name+'_'+obj_name+'.txt')
        gen_txt_labels(txt_output_path, shp_label_path, tif_map_path)
    return 'tif2png2shp2txt is done!'

def gen_labels4train(map_list, tif_dir, png_dir, obj_name, num_pos=200, num_neg=600):
    all_x_img, all_node_cat_targets, all_node_reg_targets, all_nodes_inputs, all_adj_targets, all_conn_targets, all_nodes_mask\
        = [], [], [], [], [], [], []
    
    if not os.path.exists(os.path.join(png_dir, f'{obj_name}{args.samples_folder_suffix}')):
        os.makedirs(os.path.join(png_dir, f'{obj_name}{args.samples_folder_suffix}'))
            
    sample_idx = 0
    for map_name in map_list:
        png_map_name = map_name+'.png'
        png_label_name = map_name+'_'+obj_name+'.png'
        gen_nodes_edges_targets(png_dir, tif_dir, png_map_name, png_label_name, map_name+'_'+obj_name, obj_name, num_pos, num_neg)
        