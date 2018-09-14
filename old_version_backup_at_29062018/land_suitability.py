# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:50:10 2018
Land suitability model
@author: Jing Guo
"""

from osgeo import gdal, ogr, osr
import os
from os.path import join
import pandas as pd
import shutil
import datetime as dt
import numpy as np
import sys


def reproject_raster(dataset, refdataset, outrst):
    """
    A function to reproject and resample a GDAL dataset from within 
    Python. 
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
            if geoName == 'MULTIPOLYGON':
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
    
def Array2Raster2(inArray, refRaster, newRaster):
    
    # Define NoData value of new raster
    NoData_value = float(-9999)
    
    raster = gdal.Open(refRaster)
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


def DetermineValueOrder(values):
    
    if len(values) > 1:
        if values[1] not in [-9999, -9990]:
            diff = values[1] - values[0]
            if diff > 0:
                return 'ascending'
            elif diff < 0:
                return 'descending'
            else:
                return None
        else:
            if values[1] == -9999:
                return 'descending'
            else:
                return 'ascending'
    else:
        return None

def ExtractMaxValueOfStack(array_list):
    
    suit_array_stack = np.dstack(array_list)
    # Get the index values for the minimum and maximum values
    maxIndex = np.argmax(suit_array_stack, axis=2)
#    minIndex = np.argmin(suit_array_stack, axis=2)
    
    # Create column and row position arrays
    nRow, nCol = np.shape(array_list[0])
    col, row = np.meshgrid(range(nCol), range(nRow))
    
    # Index out the maximum and minimum values from the stacked array based on row 
    # and column position and the maximum value
    maxValue = suit_array_stack[row, col, maxIndex]
#    minValue = suit_array_stack[row, col, minIndex]
    
    return maxValue

if __name__ == '__main__':
    
    prj_path = r'D:\Big_idea'
    data_path = join(prj_path, 'data')
    if not os.path.exists(data_path):
        sys.exit('The directory of cofiguration files "{}" does not exist.'.format(data_path))
    
    output_path = join(prj_path, 'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok = True)
    
    # Define NoData value of new raster
    NoData_value = -9999
    # Define resolution of output raster
    raster_res = 100
    
    tnow = dt.datetime.now()
    outSubRoot = join(output_path, 'Test_{}'.format(tnow.strftime('%Y%m%d_%H%M')))
    if os.path.exists(outSubRoot):
        shutil.rmtree(outSubRoot)
    os.makedirs(outSubRoot, exist_ok = True)
    individual_property_raster_path = join(outSubRoot, 'original_property')
    os.makedirs(individual_property_raster_path, exist_ok = True)
    
    config_file = join(data_path, 'config_2.csv')
    crop_file = join(data_path, 'crops.csv')
    
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
            outrst = join(individual_property_raster_path, '{}.tif'.format(d_a))
            refrst = p_rasters[0]
            reproject_raster(join(data_path, data_layer), refrst, outrst)    
            resample_image_Nearest(outrst, refrst)
            p_rasters.append(outrst)
            r_types.append(6)

        print('{} has been created!'.format(d_a))    
    
    # Create suitability rasters of each properties of each crops
    # iterate each crop
    for c in crops:
        subdf_crop = df_crop[df_crop['Crops']==c]
        suit_level = list(subdf_crop['Suitability'])
        
        crop_folder = join(outSubRoot, c)
        os.makedirs(crop_folder, exist_ok = True)
        
        suit_arrays = []
        
        # iterate each property
        for r, t in zip(p_rasters, r_types):
            property_array = ReadRaster(r, 1, DataTypeConversion_GDAL2NP(t))
            suit_array = np.zeros(property_array.shape)
            d_a = r.split('\\')[-1].split('.')[0] # data field name
            
            subdf_conf = df_config[df_config['Data_attr'] == d_a]
            crop_a = subdf_conf['Crop_attr']
            
            if len(crop_a) == 1:
                crop_attr_value = list(subdf_crop[crop_a.item()])
                order = DetermineValueOrder(crop_attr_value)
                
                if order is not None:
                              
                    for i in range(0, len(suit_level)):
                        value = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.item()]
                        if order == 'ascending':
                            if i == 0: 
                                suit_array = np.where(property_array<=value, suit_level[i], suit_array)
                            else:
                                value0 = subdf_crop[subdf_crop['Suitability'] == suit_level[i-1]].iloc[0][crop_a.item()]
                                if np.isnan(value) == False and (value != -9990):
                                    suit_array = np.where(np.logical_and(property_array>value0, property_array<=value), suit_level[i], suit_array)
                                else:
                                    suit_array = np.where(property_array>value0, suit_level[i], suit_array)
                                    break
                        
                        elif order == 'descending':
                            if i == 0: 
                                suit_array = np.where(property_array>=value, suit_level[i], suit_array)
                            else:
                                value0 = subdf_crop[subdf_crop['Suitability'] == suit_level[i-1]].iloc[0][crop_a.item()]
                                if (np.isnan(value) == False) and (value != -9999):
                                    suit_array = np.where(np.logical_and(property_array<value0, property_array>=value), suit_level[i], suit_array)
                                else:
                                    suit_array = np.where(property_array<value0, suit_level[i], suit_array)
                                    break
            
            elif len(crop_a) == 2:
                value = [None] * 2
                order = [None] * 2
                value0 = [None] * 2
                crop_attr_value = [None] * 2
                for i in range(0, len(crop_a)):    
                    crop_attr_value[i] = list(subdf_crop[crop_a.iloc[i]])
                    order[i] = DetermineValueOrder(crop_attr_value[i])
                    
                if (order[0] is not None) and (order[1] is not None):
                    
                    if (order[0] == 'ascending') and (order[1] == 'descending'):
                        pass
                    elif (order[0] == 'descending') and (order[1] == 'ascending'):          
                        tmp = crop_a.iloc[0]
                        crop_a.iloc[0] = crop_a.iloc[1]
                        crop_a.iloc[1] = tmp
                        tmp = None
                        
                    for i in range(0, len(suit_level)):
                        value[0] = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.iloc[0]]
                        value[1] = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.iloc[1]]
                        
                        if i == 0: 
                            if value[0] < value[1]:
                                print('Irrational values occur in {} or in {}. \nCannot generate suitability map of "{}".'.format(crop_a.iloc[0], crop_a.iloc[1], d_a))
                                break
                            suit_array = np.where(np.logical_and(property_array<=value[0], property_array>value[1]), suit_level[i], suit_array)
                        else:
                            value0[0] = subdf_crop[subdf_crop['Suitability'] == suit_level[i-1]].iloc[0][crop_a.iloc[0]]
                            value0[1] = subdf_crop[subdf_crop['Suitability'] == suit_level[i-1]].iloc[0][crop_a.iloc[1]]
                            if ((np.isnan(value[0]) == False) and (value[0] != -9990)) and ((np.isnan(value[1]) == False) and (value[1] != -9999)):
                                suit_array = np.where(np.logical_or(np.logical_and(property_array>value0[0], property_array<=value[0]), np.logical_and(property_array<=value0[1], property_array>value[1])), suit_level[i], suit_array)
                            elif ((np.isnan(value[0]) == True) or (value[0] == -9990)) and ((np.isnan(value[1]) == False) and (value[1] != -9999)):
                                suit_array = np.where(np.logical_or(property_array>value0[0], np.logical_and(property_array<=value0[1], property_array>value[1])), suit_level[i], suit_array)                  
                            elif ((np.isnan(value[0]) == False) and (value[0] != -9990)) and ((np.isnan(value[1]) == True) or (value[1] == -9999)):
                                suit_array = np.where(np.logical_or(np.logical_and(property_array>value0[0], property_array<=value[0]), property_array<=value0[1]), suit_level[i], suit_array)                                
                            else:
                                if ((np.isnan(value0[0]) == True) or (value0[0] == -9990)) and ((np.isnan(value0[1]) == False) and (value0[1] != -9999)):
                                    suit_array = np.where(property_array<=value0[1], suit_level[i], suit_array)
                                elif ((np.isnan(value0[0]) == False) and (value0[0] != -9990)) and ((np.isnan(value0[1]) == True) or (value0[1] == -9999)):
                                    suit_array = np.where(property_array>value0[0], suit_level[i], suit_array)
                                elif ((np.isnan(value0[0]) == False) and (value0[0] != -9990)) and ((np.isnan(value0[1]) == False) and (value0[1] != -9999)):
                                    suit_array = np.where(np.logical_or(property_array>value0[0], property_array<=value0[1]), suit_level[i], suit_array)
#                                break
                    
            elif len(crop_a) > 2:
                for j in range(0, len(crop_a)):    
                        
                    for i in range(0, len(suit_level)):
                        value = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.iloc[j]]
                        if np.isnan(value) == False:
                            suit_array = np.where(property_array==value, suit_level[i], suit_array)
                            break
            
            suit_array[np.where(property_array == NoData_value)] = NoData_value
            suit_raster_file = join(crop_folder, 'suit_{}.tif'.format(d_a))
            Array2Raster(suit_array, r, suit_raster_file)            
            suit_arrays.append(suit_array)
        
        crop_suit_array = ExtractMaxValueOfStack(suit_arrays)
        crop_suit_raster_file = join(crop_folder, '{}_suitability.tif'.format(c))
        Array2Raster(crop_suit_array, suit_raster_file, crop_suit_raster_file)
        
        
        
        
        
        
        
        
    