# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:10:09 2018

@author: guoj

This script is for SLMACC project. It is the second step of land suitability mapping.
It homogenizes, including resolution, extend and projection, all the covariate layers.

"""

from osgeo import gdal, ogr, osr
import os
from os.path import join
import pandas as pd
import shutil
from shutil import copyfile
import datetime as dt
import numpy as np
import sys
from os import walk


def ReprojectRaster2(raster_file, ref_raster_file, target_raster_file):
    
    raster = gdal.Open(raster_file) 
    band = raster.GetRasterBand(1)
    raster_array = band.ReadAsArray().astype(np.float)
    
    Array2Raster(raster_array, ref_raster_file, target_raster_file)
    

def reproject_raster2(dataset, refdataset, outrst):

    g = gdal.Open(dataset)
    ref = gdal.Open(refdataset)
    
    # Define spatialreference
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromWkt(g.GetProjectionRef())

    nz2000 = osr.SpatialReference()
    nz2000.ImportFromWkt(ref.GetProjectionRef())

    #Transformation function
#    tx = osr.CoordinateTransformation(wgs84,nz2000)
    
    # Get the Geotransform vector
    geo_t = g.GetGeoTransform ()
    cols = g.RasterXSize # Raster xsize
    rows = g.RasterYSize # Raster ysize
    # Work out the boundaries of the new dataset in the target projection
#    (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
#    (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*cols, geo_t[3] + geo_t[5]*rows)
    
    ulx = 992526.553934
    uly = 6199013.57078
    lrx = 2112456.44911
    lry = 4732923.52618
    
    pixelsizeX = (lrx - ulx)/cols
    pixelsizeY = (lry - uly)/rows

    # See how using 27700 and WGS84 introduces a z-value!
    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('GTiff')
    # The size of the raster is given the new projection and pixel spacing
    # Using the values we calculated above. Also, setting it to store one band
    # and to use Float32 data type.
    dest = mem_drv.Create(outrst, cols, rows, 1, gdal.GDT_Float32)
    # Calculate the new geotransform
    new_geo = (ulx, pixelsizeX, geo_t[2], uly, geo_t[4], pixelsizeY)

    # Set the geotransform
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(nz2000.ExportToWkt())
    
    #set No data value
    outband = dest.GetRasterBand(1)
    outband.SetNoDataValue(-9999)
    # Perform the projection/resampling 
    gdal.ReprojectImage(g, dest, wgs84.ExportToWkt(), nz2000.ExportToWkt(), gdal.GRA_NearestNeighbour)

def reproject_raster(dataset, refdataset, outrst):
    """
    A function to reproject and resample a GDAL dataset from within 
    Python. 
    
    For some raster, this function has some unexpected issues on 'TransformPoint',
    which result in the incorrect extent of projected raster.
    
    An alternative way is to use 'reproject_raster2' function
    which set fixed values of target raster.(Can create from ArcGIS)
    """

    g = gdal.Open(dataset)
    ref = gdal.Open(refdataset)
    
    # Define spatialreference
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromWkt(g.GetProjectionRef())

    nz2000 = osr.SpatialReference()
    nz2000.ImportFromWkt(ref.GetProjectionRef())

    #Transformation function
    tx = osr.CoordinateTransformation(wgs84,nz2000)
    
    # Get the Geotransform vector
    geo_t = g.GetGeoTransform ()
    cols = g.RasterXSize # Raster xsize
    rows = g.RasterYSize # Raster ysize
    # Work out the boundaries of the new dataset in the target projection
    (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
    (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*cols, geo_t[3] + geo_t[5]*rows)
    
    pixelsizeX = (lrx - ulx)/cols
    pixelsizeY = (lry - uly)/rows

    # See how using 27700 and WGS84 introduces a z-value!
    # Now, we create an in-memory raster
    mem_drv = gdal.GetDriverByName('GTiff')
    # The size of the raster is given the new projection and pixel spacing
    # Using the values we calculated above. Also, setting it to store one band
    # and to use Float32 data type.
    dest = mem_drv.Create(outrst, cols, rows, 1, gdal.GDT_Float32)
    # Calculate the new geotransform
    new_geo = (ulx, pixelsizeX, geo_t[2], uly, geo_t[4], pixelsizeY)

    # Set the geotransform
    dest.SetGeoTransform(new_geo)
    dest.SetProjection(nz2000.ExportToWkt())
    
    #set No data value
    outband = dest.GetRasterBand(1)
    outband.SetNoDataValue(-9999)
    # Perform the projection/resampling 
    gdal.ReprojectImage(g, dest, wgs84.ExportToWkt(), nz2000.ExportToWkt(), gdal.GRA_NearestNeighbour)

#    return dest

def resample_image_Nearest(dataset, refdataset):
    
    
    r_fine = gdal.Open(refdataset) 
    r_coarse = gdal.Open(dataset) 

    band = r_coarse.GetRasterBand(1)
    coarse_array = band.ReadAsArray().astype(np.float)
    
    #---------------------------------------------------------------
    #grow coarse raster for 1 pixel based on the nearest value
    cols = r_coarse.RasterXSize
    rows = r_coarse.RasterYSize
    growArray = np.full((rows, cols), float(NoData_value))
    for x in range(0, rows):
        for y in range(0, cols):
            if int(coarse_array[x, y]) == NoData_value:
                neighbors = []
                for i in range(x-1, x+2):
                    for j in range(y-1, y+2):
                        try:
                            if int(coarse_array[i,j]) != NoData_value:
                                neighbors.append(coarse_array[i,j])
                        except:
                            pass
                if len(neighbors) > 0:
                    growArray[x, y] = np.mean(neighbors)
     
    coarse_array = np.where(coarse_array == NoData_value, growArray, coarse_array)     
    #---------------------------------------------------------------
                        
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r_fine.GetGeoTransform()          #spatial reference of fine resolution raster        	
    (upper_left_x1, x_size1, x_rotation1, upper_left_y1, y_rotation1, y_size1) = r_coarse.GetGeoTransform()   #spatial reference of coarse resolution raster
    
    xlimit=r_fine.RasterXSize        	
    ylimit=r_fine.RasterYSize

    rel_pos_x=np.zeros(xlimit,dtype=np.int32)
    rel_pos_y=np.zeros(ylimit,dtype=np.int32)

    for x_index in range(0,xlimit):
        x_coords = x_index * x_size + upper_left_x + (x_size / 2) # converts index to coordinatex_coords1    		
        rel_pos_x[x_index] = np.int((x_coords - upper_left_x1) / x_size1)     #converts coordinate back to index in coarse resolution data

    for y_index in range(0,ylimit):
        y_coords = y_index * y_size + upper_left_y + (y_size / 2) # converts index to coordinate
        rel_pos_y[y_index] = int((y_coords - upper_left_y1) / y_size1)     #converts coordinate back to index in coarse resolution data



    xlimit=len(rel_pos_x)
    ylimit=len(rel_pos_y)

    newArray = np.zeros((ylimit,xlimit))

    for x_offset in range(0,xlimit):
        for y_offset in range(0,ylimit):
            newArray[y_offset,x_offset] = coarse_array[rel_pos_y[y_offset],rel_pos_x[x_offset]]

    r_coarse = None
    Array2Raster(newArray, refdataset, dataset)

def CompareRasterSize(raster, refraster):
    r = gdal.Open(raster) 
    r_ref = gdal.Open(refraster) 

    r_cols = r.RasterXSize
    r_rows = r.RasterYSize
    
    r_ref_cols = r_ref.RasterXSize
    r_ref_rows = r_ref.RasterYSize
    
    if (r_cols != r_ref_cols) or (r_rows != r_ref_rows):
        resample_image_Nearest(raster, refraster)
    else:
        pass

def RemoveEmptyRowsCols(np_array):
    rows = np_array.shape[0]
    cols = np_array.shape[1]
    
    for row in range(0, rows):
        print(row)
        print(set(np_array[row,:]))
        if row > rows:
            break
        if len(set(np_array[row,:])) <= 1:   # Caution!!! this condition is not strong enough
            np_array = np.delete(np_array, row, axis=0)
            rows-=1
    
    for col in range(0, cols):
        if len(set(np_array[:,col])) <= 1:   # Caution!!! this condition is not strong enough
            np_array = np.delete(np_array, col, axis=1)
            cols-=1
    return np_array        
    
def ESRIVector2Raster(inPath, inFile, fileType, field_Name):
    
    if fileType == 'featureclass':
        driver = ogr.GetDriverByName("OpenFileGDB")
        ds = driver.Open(inPath, 0)
        layer = ds.GetLayer(inFile)
    elif fileType == 'shapefile':
        driver = ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.Open(join(inPath, inFile), 0)
        layer = ds.GetLayer()  
    else:
        layer = None
    
    if layer is not None:
        feature = layer.GetNextFeature()
        geometry = feature.GetGeometryRef()
        geoName = geometry.GetGeometryName()
        fieldType = GetLayerFieldType(layer, field_Name)
        

        if fieldType is not None:
            if geoName == 'MULTIPOLYGON' or geoName == 'POLYGON':
                out_raster_file = join(individual_property_raster_path, '{}.tif'.format(field_Name))
                PolygonToRaster(layer, out_raster_file, field_Name, raster_res, DataTypeConversion(fieldType))
                return out_raster_file, DataTypeConversion(fieldType)
            else:
                print('"{}" is not a polygon layer!'.format(inFile))
        else:
            print('Field "{}" does not exist in layer "{}"!'.format(field_Name, inFile))
    else:
        print('Layer "{}" does not exist!'.format(inFile))
        

def GetLayerFieldType(layer, field_name):
    lyrDefn = layer.GetLayerDefn()
    for i in range(lyrDefn.GetFieldCount()):
        fieldName =  lyrDefn.GetFieldDefn(i).GetName()
        if fieldName == field_name:
            fieldTypeCode = lyrDefn.GetFieldDefn(i).GetType()
            fieldType = lyrDefn.GetFieldDefn(i).GetFieldTypeName(fieldTypeCode)
            break
        else:
            fieldType = None
            
    return fieldType


# Convert polygon shapefile to tif
def PolygonToRaster(inLayer, outFile, attr, pixel_size, d_type):
    
    # Open the data source and read in the extent

    srs=inLayer.GetSpatialRef()
    x_min, x_max, y_min, y_max = inLayer.GetExtent()

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)
    
    target_ds = gdal.GetDriverByName('GTiff').Create(outFile, x_res, y_res, 1, d_type)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    target_ds.SetProjection(srs.ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], inLayer, options = ["ATTRIBUTE="+attr])
    
#    print('{} {}'.format(attr, d_type))
    inLayer = None
    target_ds = None
    band = None

def DataTypeConversion(x):
    return {
            'Real': gdal.GDT_Float32,
            'Integer': gdal.GDT_Int32,
        }.get(x, gdal.GDT_Unknown)
    
def DataTypeConversion_GDAL2NP(x):
    return {
            5: np.int,
            6: np.float,
        }.get(x, np.float)

def ReadRaster(inRaster, band, dataType):
    
    source = gdal.Open(inRaster)
    band = source.GetRasterBand(band)
    valArray = band.ReadAsArray().astype(dataType)
    
    return valArray

def Array2Raster(inArray, refRaster, newRaster):
    
    # Define NoData value of new raster
    NoData_value = float(-9999)
    
    raster = gdal.Open(refRaster)
    refband = raster.GetRasterBand(1)
    refArray = refband.ReadAsArray().astype(np.float)
    
    inArray = np.where(refArray==NoData_value, NoData_value, inArray)
    
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = inArray.shape[1]
    rows = inArray.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRaster, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)
    outband.WriteArray(inArray)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
    
if __name__ == '__main__':
    
    prj_path = r'D:\LandSuitability_AG'
    configfile_path = join(prj_path, 'data', 'config')
    if not os.path.exists(configfile_path):
        sys.exit('The directory of cofiguration files "{}" does not exist.'.format(configfile_path))
    
    output_path = join(prj_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok = True)
    
    climate_reference_raster = join(prj_path, 'data\climate\sr_nz200_ref', 'climate_nz2000.tif')
    # Define NoData value of new raster
    NoData_value = -9999
    # Define resolution of output raster
    raster_res = 500
    
    tnow = dt.datetime.now()
    outSubRoot = join(output_path, 'Test_{}'.format(tnow.strftime('%Y%m%d_%H%M')))
    if os.path.exists(outSubRoot):
        shutil.rmtree(outSubRoot)
    os.makedirs(outSubRoot, exist_ok = True)
    individual_property_raster_path = join(outSubRoot, 'original_property')
    os.makedirs(individual_property_raster_path, exist_ok = True)
    
    config_file = join(configfile_path, 'config.csv')
    crop_file = join(configfile_path, 'crops.csv')
    
    df_config = pd.read_csv(config_file, thousands=',')
    df_crop = pd.read_csv(crop_file, thousands=',')
    
    data_attr = list(df_config['Data_attr'].dropna().unique())
    
    crops = list(df_crop['Crops'].dropna().unique())
    
    # Preprocessing, including rasterize smap polygon and resample the climate data
    p_rasters = []
    r_types = []
    iraster = 0
    for d_a in data_attr:
        records = df_config[df_config['Data_attr'] == d_a]

        data_type = records.iloc[0]['Data_type']
        data_path = records.iloc[0]['Data_path']
        data_layer = records.iloc[0]['Data_layer']

                
        if (data_type == 'featureclass') or (data_type == 'shapefile'):
            property_raster, data_t = ESRIVector2Raster(data_path, data_layer, data_type, d_a)
            p_rasters.append(property_raster)
            r_types.append(data_t)

        elif data_type == 'raster':
            # If the raster data layer name ends with '_' 
            # indicates this is a climate layer which has a time series of data to deal with.
            if data_layer.split('.')[0][-1] == '_':
                files = []
                for (root, subroot, filenames) in walk(data_path):
                    files.extend(filenames)
                
                for f in files:
                    if data_layer.split('.')[0] in f:
                        outrst = join(individual_property_raster_path, f)
                        if not p_rasters[0]:
                            refrst = join(data_path, f)
                        else:
                            refrst = p_rasters[0]
#                        ReprojectRaster2(join(data_path, f), climate_reference_raster, outrst)
                        reproject_raster2(join(data_path, f), refrst, outrst)
#                        copyfile(join(data_path, f), outrst)
                        p_rasters.append(outrst)
                        r_types.append(6)
                        
            else: # Otherwise just one raster to deal with.
                outrst = join(individual_property_raster_path, '{}.tif'.format(d_a))
                if not p_rasters[0]:
                    refrst = join(data_path, data_layer)
                else:
                    refrst = p_rasters[0]
                reproject_raster2(join(data_path, data_layer), refrst, outrst)
#                copyfile(join(data_path, data_layer), outrst)
                p_rasters.append(outrst)
                r_types.append(6)

        print('{} has been created!'.format(d_a)) 
    
    # Resample rasters to make sure they are consistent with the first created raster in resolution and extent.    
    print('Start to resample raster files...')
    
    i = 0
    for r in p_rasters:
        if i > 0:

            resample_image_Nearest(r, p_rasters[0])
    
        i+=1
        
        
        