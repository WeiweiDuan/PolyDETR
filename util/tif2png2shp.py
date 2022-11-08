import cv2
import os, sys
import numpy as np
sys.path.append(".")
from .process_shp import read_shp

def tif2png(tif_path, png_path):
    print('tif path: ', tif_path)
    tif_map = cv2.imread(tif_path)
    print('tif map shape: ', tif_map.shape)
    if np.max(tif_map) == 1:
        print('label image {0, 1}')
        tif_map = tif_map*255
    cv2.imwrite(png_path, tif_map)
    return 'save png map in %s'%png_path

def gen_txt_labels(save_path, shp_path, tif_path):
    polylines = read_shp(shp_path, tif_path)
    nodes = list()
    for line in polylines:
        for p in line:
            nodes.append(p)
    nodes_np = np.array(nodes)
    np.savetxt(save_path, nodes_np, fmt='%d', delimiter=',')
    return 'save nodes in %s'%save_path