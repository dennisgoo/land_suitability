# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:40:26 2018

@author: 
    
This script is for SLMACC project. It is the third step of land suitability mapping.
It creates land suitability maps for each crop of each contributing covariate layer, 
as well as an overall suitability map for each crop.    

"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:50:10 2018
Land suitability model
@author: Jing Guo
"""

from osgeo import gdal, osr
import os
from os.path import join
import pandas as pd
import numpy as np
import sys
from os import walk


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


def homogenize_nodata_area(array_list, NoData):
    '''
    This function is used to make create a mask array, 
    and its' Nodata grids match all the Nodata grids in 
    each array in the array list. 
    '''
    for i in range(0, len(array_list)):
        if i == 1:
            exp = np.logical_and(array_list[i]!= NoData, array_list[i-1]!= NoData,)
        elif i > 1:
            exp = np.logical_and(exp, array_list[i]!= NoData)
    
    ref_array = np.where(exp, array_list[0], NoData)
    return ref_array


def strip_end(text, suffix):
    if not text.endswith(suffix):
        return text
    return text[:len(text)-len(suffix)]


def suitability_mapping(df_config, df_crop, coviriate_raster_list):
    '''
    Core function of suitability mapping.
    'df_config' - panda dataframe of 'config.csv' file
    'df_crop' - panda dataframe of 'crop.csv' file
    'coviriate_raster_list' - a list of all the coviriate layers (directory plus filename)
    '''
    
    # get a crop list   
    crops = list(df_crop['Crops'].dropna().unique())
   
    # Create suitability rasters of each properties of each crops
    # iterate each crop
    for c in crops:
        subdf_crop = df_crop[df_crop['Crops']==c]    # dataframe of one crop - c
        suit_level = list(subdf_crop['Suitability']) # the suitability level of crop c
        
        crop_folder = join(outSubRoot, c)            # output folder of suitability maps of crop c
        os.makedirs(crop_folder, exist_ok = True)
        
        # climatic layer can be in a time series, and when calculate the overall suitability, they should be seperated,
        # so here use two lists to distinguish climatic and non-climatic layers
        non_climatic_arrays = []
        climatic_arrays = []
        years = []   # a list of years of the climatic layers
        
        # iterate each coviriate layer
        for r in coviriate_raster_list:
            property_array = ReadRaster(r, 1, np.float)  # an array of the coviriate
            suit_array = np.zeros(property_array.shape)  # an empty array for suitability array (fill with value afterwards)
            d_a = r.split('\\')[-1].split('.')[0]        # data attribute name, consistent with the  "Data_attr" value in config.csv
                                                         # this velue is extracted from the filename of coviriate raster. 
                                                         # When these rasters were created in the 'covireate_preprocessing.py' script, 
                                                         # their name were set based on the "Data_attr" values.   
            suit_raster_file = join(crop_folder, 'suit_{}.tif'.format(d_a)) # raster filename of suitability map of crop c

            # When dealing with the climatic rasters the 'year' number need to be eliminated,
            # to keep the consistency of the "Data_attr" values in config.csv.
            # 'annual' here is the key word to determine wether the coviriate raster is climatic or not,
            # and this is based on the coviriate rasters's filename. 
            # BE CAUTION!!! If the filename were changed,
            # e.g. when monthly raster are created, this key words will not work, it should be changed to 'monthly'
            # or whatever that can differenciate the climatic raster from the others.
            if 'annual' in d_a                           
                years.append(d_a.split('_')[-1])
                d_a = strip_end(d_a, d_a.split('_')[-1])
                                
            
            subdf_conf = df_config[df_config['Data_attr'] == d_a] # a sub-dataframe of config file dataframe which only has records of coviriate 'd_a'
            crop_a = subdf_conf['Crop_attr']                      # extract the crop attributes 
            
            # the number of crop attribute is used to determine the type of coviriate, either continual or catorgrical,
            # as well as to determine, when it is a continual coviriate, wether it is a one direction criterion (either ascending or descending) 
            # or a two directions criterion (both ascending and descending). 
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
            Array2Raster(suit_array, r, suit_raster_file)            
        
            if 'annual' in d_a:
                climatic_arrays.append(suit_array)
            else:
                non_climatic_arrays.append(suit_array)
        
        to_be_stacked_arrays = []
        i = 0
        for sa in climatic_arrays:

            to_be_stacked_arrays = []
            to_be_stacked_arrays.extend(non_climatic_arrays)
            to_be_stacked_arrays.append(sa)
    
            crop_suit_array = ExtractMaxValueOfStack(to_be_stacked_arrays)
            ref_array = homogenize_nodata_area(to_be_stacked_arrays, NoData_value)
            crop_suit_array[np.where(ref_array == NoData_value)] = NoData_value
            crop_suit_raster_file = join(crop_folder, '{}_suitability_{}.tif'.format(c, years[i]))
            Array2Raster(crop_suit_array, suit_raster_file, crop_suit_raster_file)
            i += 1

if __name__ == '__main__':
    
    config_path = r'D:\LandSuitability_AG\data\config'     # the path of the two configration files
    output_path = r'D:\LandSuitability_AG\output'          # the path of output rasters 
    data_path = join(output_path, 'original_property')     # the path of coviriate layers 
    
    if not os.path.exists(data_path):
        sys.exit('The directory of cofiguration files "{}" does not exist.'.format(data_path))
    
    # create a directory for suitability maps
    outSubRoot = join(output_path, 'suitability_maps1')
    if not os.path.exists(outSubRoot):
        os.makedirs(outSubRoot, exist_ok = True)
    
    # Set NoData value of new raster
    NoData_value = -9999

    
    config_file = join(config_path, 'config.csv')
    crop_file = join(config_path, 'crops.csv')
    
    df_config = pd.read_csv(config_file, thousands=',')
    df_crop = pd.read_csv(crop_file, thousands=',')
    
    
    # get all the coviriate tif layers
    p_rasters = []
    for (root, subroot, filenames) in walk(data_path):
        for f in filenames:
            if f.split('.')[-1].lower() in ['tif', 'tiff']:
                p_rasters.append(join(root, f))

        break
    
    
    suitability_mapping(df_config, df_crop, p_rasters)
        
        
        
        
        
        
        
        
    