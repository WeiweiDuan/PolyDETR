import os
import sys
import json
import argparse
from util.args import get_args_parser
from util.args_test_multimaps import get_args_parser
from test_one_map import test_single_map
from util.util_helper import create_folder

# evaluation map names
map_list = ['RI_Uxbridge', 'SD_BlackHills', 'USCan_LakeSuperior', 'MT_RedRockLakes', 'VA_Stanardsville', 'VA_Hayfield', 'pp1410b']

single_map_obj_list = ['caldera_margin_line', 'crest_line_moraine_line', 'crest_moraine_line', \
                  'diabase_line', 'dike_sandstone_line', 'dune_crest_line', 'erosional_scarp_line',\
                   'fluvial_scarp_line', 'fracture_line', 'glacial_moraine_line', 'fault_reverse_line', \
                  'gravel_pit_line', 'intra-sequence_thrust_fault_line', 'kg_granitic_dike_line', 'quartz_vein_line', 'lava_flow_contact_line', 'major_sequence-bounding_thrust_fault_line', 'toe_solifluc_deposit_line', 'pressure_buckle_line', 'fault_offshore_line', 'lava_pond_line', 'quartz_veins_line', 'low_angle_fault_line' ]

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])
args.gpu = '1'
args.pred_dir = '/data/weiweidu/data/final_submission'
create_folder(args.png_map_dir)
create_folder(args.pred_dir)

for map_name in map_list:
    json_path = os.path.join(args.tif_map_dir, map_name+'.json')
    json_file = open(json_path)
    metadata = json.load(json_file)
    args.map_name = map_name
    for symbol in metadata['shapes']:
        if '_line' not in symbol['label']:
            continue
        args.obj_name = symbol['label'].lower()
        empty_flag = False
        print('----- processing {}_{} -------'.format(args.map_name, args.obj_name))
        if 'thrust_fault_line' == symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_final/trained_models/thrust_fault_line_img512_g32_e330.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 512
            args.dilation = False
            args.grid_size = 32
            args.num_nodes = 200
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
        elif 'erosional_scarp_line'== symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/erosional_scarp_line.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 512
            args.dilation = False
            args.grid_size = 32
            args.num_nodes = 200
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
        elif 'fluvial_scarp_line'== symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/fluvial_scarp_line.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 512
            args.dilation = False
            args.grid_size = 32
            args.num_nodes = 200
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
        elif 'scarp_line' in symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/landslide_scarp_line_moreepoch.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
        elif 'lineament' in symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/lineament_line_final.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
        elif 'fault_line' == symbol['label'].lower():
            args.trained_model_path ='/data/weiweidu/PolyDETR_final/trained_models/fault_line_img512_g32_all_color_aug_final.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 512
            args.dilation = False
            args.grid_size = 32
            args.num_nodes = 200
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
        elif symbol['label'].lower() in single_map_obj_list:
            args.trained_model_path = \
                os.path.join('/data/weiweidu/PolyDETR_load_img_from_folder/trained_models', symbol['label'].lower()+'.pth')
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 512
            args.dilation = False
            args.grid_size = 32
            args.num_nodes = 200
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)  
        elif 'reverse_fault_line' == symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/reverse_fault_line_final.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 512
            args.dilation = False
            args.grid_size = 32
            args.num_nodes = 200
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args) 
        else:
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/hi_angle_fault_line_final.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
        if empty_flag:
            args.trained_model_path = '/data/weiweidu/PolyDETR_final_copy1/trained_models/fault_line_img256_g16_e330.pth'
            if not os.path.exists(args.trained_model_path):
                print(f'{args.trained_model_path} !!!!!!!!!!!!!')
                continue
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            print(f'----- {map_name}_{args.obj_name} using {args.trained_model_path} -----')
            empty_flag = test_single_map(args)
