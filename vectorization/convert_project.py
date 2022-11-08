from osgeo import ogr
import os
from osgeo import gdal
from osgeo.gdalconst import *
import subprocess
import math
import numpy as np

def read_shp(ds_path, map_path=None):
    ds = ogr.Open(ds_path)
    layer = ds.GetLayer(0)
    f = layer.GetNextFeature()

    count = 0
    lines = []
    while f:
        geom = f.GetGeometryRef()
        l = []
        if geom != None:
            # points = geom.GetPoints()
            points = geom.ExportToJson()
            points = eval(points)['coordinates']
            print(points)
            lines.append(points)
        count += 1
        f = layer.GetNextFeature()
    return lines

# if __name__ == '__main__':
#     read_shp('C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_100414_2001_24000_bag\data\Perfect_shp\CA_Bray_waterlines_2001_perfect.shp')
