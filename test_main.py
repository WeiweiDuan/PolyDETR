import os
import sys
import json
import argparse
from util.args import get_args_parser
from util.args_test_multimaps import get_args_parser
from test_one_map import test_single_map
from util.util_helper import create_folder

# validation set
# map_list = ['NV_HiddenHills', 'Genesis_fig9_4', 'AK_Dillingham', 'MN', 'AR_StJoe', 'AZ_PrescottNF', 'AZ_GrandCanyon', 'OR_Carlton', 'NM_Sunshine', 'DMEA2328_OldLeydenMine_CO', '24_Black Crystal_2014_11', 'Trend_2005_fig7_2', 'NM_Volcanoes', 'KY_WestFranklin', 'OR_Camas', '26_Black Crystal_2018_6', 'AK_Hughes', 'AK_Kechumstuk', 'Genesis_fig9_5', 'CA_NV_DeathValley', 'Genesis_fig7_1', 'Genesis_fig9_1', 'Genesis_fig9_2', 'CA_Elsinore', 'Genesis_fig9_3', 'Trend_2005_fig6_4', 'CO_Alamosa', 'USGS_B-961_6', '46_Coosa_2015_11 74', 'AZ_PipeSpring', 'AR_Maumee']

map_list = ['AK_Noatak,AR_Jasper,AZ_CA_CastleRock,AZ_CA_Topock']

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])
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
        print('----- processing {}_{} -------'.format(args.map_name, args.obj_name))
        if 'fault_line' in symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder_copy2/trained_models/fault_line_all_final.pth'
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            test_single_map(args)
        elif 'thrust_fault_line' in symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder_color_aug_bbox/trained_models/thrust_fault_line_color_aug_bbox_all_e210.pth'
            args.img_size = 512
            args.dilation = False
            args.grid_size = 32
            args.num_nodes = 200
            test_single_map(args)
        elif 'scarp_line' in symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/landslide_scarp_line_moreepoch.pth'
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            test_single_map(args)
        elif 'lineament' in symbol['label'].lower():
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder/trained_models/lineament_line_final.pth'
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            test_single_map(args)
        else:
            args.trained_model_path = '/data/weiweidu/PolyDETR_load_img_from_folder_copy2/trained_models/fault_line_all_final.pth'
            args.img_size = 256
            args.dilation = True
            args.grid_size = 16
            args.num_nodes = 150
            test_single_map(args)