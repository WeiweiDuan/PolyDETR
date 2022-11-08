import argparse
"""
questionable maps for thrust fault lines
AK_Seldovia (label named as AK_Seldovia_Thrust_fault_line),
CA_Cambria (no thrust fault line in label image),
"""
def get_args_parser():
    parser = argparse.ArgumentParser('Set PolyDETR', add_help=False)
#     add -f to avoid the error
    parser.add_argument('-f', default=None, type=str)
    
    # Training data
    parser.add_argument('--obj_name', default='thrust_fault_line', type=str,
                       help='desired object name')
    parser.add_argument('--map_name_list', default='CO_Frisco,CA_BartlettSprings,ID_basement,CO_Bailey,AZ_PeachSprings,CA_WhiteLedgePeak,AK_HowardPass,AK_Seldovia,VA_Lahore_bm,CA_NV_LasVegas,AK_LookoutRidge,CO_ClarkPeak,AK_Christian,AK_Ikpikpuk,CO_Elkhorn,CO_GreatSandDunes', type=str,
                       help='multiple map names, separated by comma')  
    parser.add_argument('--input_data_dir', default='/data/weiweidu/data/training', type=str,
                       help='the dir to tif map and label images')
    parser.add_argument('--output_data_dir', default='/data/weiweidu/data/training_thesis', type=str,
                       help='the dir to save png map and shapfiles for labels')
    parser.add_argument('--samples_folder_suffix', default='_samples4thesis_img512_grid32', type=str,
                       help='training samples folder named as obj_name, if it has variation, add suffix')
    parser.add_argument('--img_size', default=512, type=float,
                       help='size of a training image')
    # Training settings
    parser.add_argument('--num_pos_samples', default=200, type=int,
                       help='the number of images covering desired polyline')
    parser.add_argument('--num_neg_samples', default=350, type=int,
                       help='the number of images not covering desired polyline')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_drop', default=150, type=int)
    parser.add_argument('--epochs', default=420, type=int)
    parser.add_argument('--saved_model_dir', default='./trained_models',
                        help='path where to save, empty for no saving')
    parser.add_argument('--saved_model_name', default='fault_line_img512_g32.pth',
                       help='model name saved after training') 
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu', default='1', help='gpu name')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')
    parser.add_argument('--trained_model_path', default=None,
                       help='load the pre-trained model from the path')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use") #resnet50
    parser.add_argument('--dilation', default=False, action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--masks', default=False, action='store_true',
                        help="Train segmentation head for the backbone")

    # * PolyDETR hyperparams
    parser.add_argument('--grid_size', default=32, type=int,
                       help="size of grid, one grid cell in a cropped image proposes one candidate node")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks") #2048
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)") #256
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer") #was 0.1
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    parser.add_argument('--pre_norm', default=True, action='store_true')
    parser.add_argument('--num_nodes', default=200, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', default=False, dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # Loss coefficients
    parser.add_argument('--eos_coef', default=50.0, type=float,
                        help="the weight for the node regression branch")

    return parser
