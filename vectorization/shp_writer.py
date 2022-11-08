import os
import sys
from osgeo import ogr, gdal, osr
sys.path.append(".")
from .convert_project import read_shp
import warnings
warnings.filterwarnings("ignore")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def pixel2coord(coor_path, x, y):
    raster = gdal.Open(coor_path)
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

def coor2epsg(coor_path, epsg, x, y):
    # get CRS from dataset
    ds = gdal.Open(coor_path)
    crs = osr.SpatialReference()
    # crs.ImportFromProj4(ds.GetProjection())
    crs.ImportFromWkt(ds.GetProjectionRef())
# create lat/long crs with WGS84 datum
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(epsg)
    t = osr.CoordinateTransformation(crs, crsGeo)
    (lat, long, z) = t.TransformPoint(x, y)
    return (lat, long)

def createShapefile(shapefileName, line, coor_path, epsg=None, append=False):
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Creating a new data source and layer
    if os.path.exists(shapefileName) and not append:
        driver.DeleteDataSource(shapefileName)
    if epsg == None:
        srs = gdal.Open(coor_path).GetProjection()
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromProj4(srs)
    else:
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(epsg)

    if append:
        ds = driver.Open(shapefileName, 1)
        layer = ds.GetLayer()
    else:
        ds = driver.CreateDataSource(shapefileName)
        layer = ds.CreateLayer('layerName', spatial_reference, geom_type = ogr.wkbLineString)

    if ds is None:
        print ('Could not create file')
        sys.exit(1)

    # add a field to the output
    fieldDefn = ogr.FieldDefn('fieldName', ogr.OFTReal)
    layer.CreateField(fieldDefn)
    cnt = 0
    for v in line:
        cnt += 1
        lineString = ogr.Geometry(ogr.wkbLineString)
        if len(v) == 1:
            print('line only has one point!')
            continue
        for m in v:
            if type(m) != list:
                p = m.split(' ')
                if not is_number(p[0]):
                    continue
                if len(p) < 2:
                    continue
            else:
                p = m
#             print(p)
            x, y = float(p[0]), float(p[1])
            # lineString.AddPoint(x, -y)
            xp, yp = pixel2coord(coor_path,x,y)
            if epsg != None:
                xp, yp = coor2epsg(coor_path, epsg, xp, yp)
            lineString.AddPoint(xp,yp)

        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(lineString)
        feature.SetField('fieldName', 'LineString')
        layer.CreateFeature(feature)
        lineString.Destroy()
        feature.Destroy()
    ds.Destroy()
    print ("Shapefile created")


# if __name__ == '__main__':
    # lines = read_shp('C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_100414_2001_24000_bag\data\Perfect_shp\CA_Bray_waterlines_2001_perfect.shp')
    # DATA_DIR = 'C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_100414_2001_24000_bag'
    # COOR_SYS_PATH = os.path.join(DATA_DIR, 'CA_Bray_100414_2001_24000_geo.tif')
    # shp_name = 'C:\Users\weiweiduan\Documents\Map_proj_data\CA\CA_Bray_100414_2001_24000_bag\data\Perfect_shp\CA_Bray_waterlines_2001_perfect_epsg4267.shp'
    # createShapefile(shp_name, lines, COOR_SYS_PATH, epsg=4267)
