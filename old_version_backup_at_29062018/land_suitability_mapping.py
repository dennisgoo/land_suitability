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
import CSVOperation
import datetime as dt

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
        # the -9999 is a end mark of descending attribution while the -9990 is an ascending attribute end mark
        # these marks are set in 'crop.csv' file in case there is just one vaule in any continual attribute,
        # to help the script konw the order of the attribute
        
        for v in values:
            if v == -9999:
                return 'descending'
            elif v == -9990:
                return 'ascending'

        for i in range(0, len(values)-1):
            diff = values[i+1] - values[i]
            if diff > 0:
                return 'ascending'
            elif diff < 0:
                return 'descending'
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
    maxValue_array = suit_array_stack[row, col, maxIndex]
#    minValue = suit_array_stack[row, col, minIndex]
    
    return maxValue_array

def ExtractMaxValueIndexBinaryOfStack(array_list, name_list, max_value):
    
    def getNameGroup(binary_string, name_list):
        
        binary_string = binary_string[2:] # Remove the first two charactors '0b'
        
        name_group = []
        
        for i in range(0, len(binary_string)):
            if binary_string[i] == '1':
                name_group.append(name_list[i])
        
        return name_group

    
    max_index_binary_array = '0b'
    max_count_array = np.zeros(array_list[0].shape)
    
    for array in array_list:
        max_index_array = np.where(array == max_value, '1', '0')
        max_index_binary_array = np.core.defchararray.add(max_index_binary_array, max_index_array)
        
        max_array = np.where(array == max_value, 1, 0)
        max_count_array = max_count_array + max_array
        
        
    max_index_int_array = np.zeros(max_index_binary_array.shape)
    for i in range(0, len(max_index_binary_array)):
        for j in range(0, len(max_index_binary_array[i])):
            max_index_int_array[i][j] = int(max_index_binary_array[i][j], 2)
    
    
    unique_binary_list = list(np.unique(max_index_binary_array))
    
    max_value_legend_list = []
    
    for binary in unique_binary_list:
        
        key = int(binary, 2)
        group_name = getNameGroup(binary, name_list)
        max_value_legend_list.append([key, len(group_name), '&'.join(group_name)])
        
    return max_index_int_array, max_count_array, max_value_legend_list


def homogenize_nodata_area(array_list, NoData):
    '''
    This function is used to create a mask array, 
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
        print('create {} suitability raster at {}...'.format(c, dt.datetime.now()))
        subdf_crop = df_crop[df_crop['Crops']==c]    # dataframe of one crop - c
        suit_level = list(subdf_crop['Suitability']) # the suitability level of crop c
        
        crop_folder = join(outSubRoot, c)            # output folder of suitability maps of crop c
        os.makedirs(crop_folder, exist_ok = True)
        
        # climatic layer can be in a time series, and when calculate the overall suitability, they should be seperated,
        # so here use two lists to distinguish climatic and non-climatic layers
        
        suit_arrays = []
#        non_climatic_arrays = []
#        climatic_arrays = []
        years = []   # a list of years of the climatic layers
        covariate_list = [] # a list of covariates name
        # iterate each coviriate layer
        for r in coviriate_raster_list:
            property_array = ReadRaster(r, 1, np.float)  # an array of the coviriate
            suit_array = np.zeros(property_array.shape)  # an empty array for suitability array (fill with value afterwards)
            d_a = r.split('\\')[-1].split('.')[0]        # data attribute name, consistent with the  "Data_attr" value in config.csv
            covariate_list.append(d_a)                   # this velue is extracted from the filename of coviriate raster. 
                                                         # When these rasters were created in the 'covireate_preprocessing.py' script, 
                                                         # their name were set based on the "Data_attr" values.   
            suit_raster_file = join(crop_folder, '{}_suit_{}.tif'.format(c, d_a)) # raster filename of suitability map of crop c

            # When dealing with the climatic rasters the 'year' number need to be eliminated,
            # to keep the consistency of the "Data_attr" values in config.csv.
            # 'annual' here is the key word to determine wether the coviriate raster is climatic or not,
            # and this is based on the coviriate rasters's filename. 
            # BE CAUTION!!! If the filename were changed,
            # e.g. when monthly raster are created, this key words will not work, it should be changed to 'monthly'
            # or whatever that can differenciate the climatic raster from the others.
            if 'annual' in d_a:                           
                years.append(d_a.split('_')[-1])
                d_a = strip_end(d_a, d_a.split('_')[-1])
                                
            
            subdf_conf = df_config[df_config['Data_attr'] == d_a] # a sub-dataframe of config file dataframe which only has records of coviriate 'd_a'
            crop_a = subdf_conf['Crop_attr']                      # extract the crop attributes 
            
            # the number of crop attribute is used to determine the type of coviriate, either continual or categorical,
            # as well as to determine, when it is a continual coviriate, wether it is a one direction criterion (either ascending or descending) 
            # or a two directions criterion (both ascending and descending). 
            if len(crop_a) == 1:
                crop_attr_value = list(subdf_crop[crop_a.item()])
                order = DetermineValueOrder(crop_attr_value)   # DetermineValueOrder function is used to check whether the current crop attribute is ascending or descending
                
                if order is not None:
                              
                    for i in range(0, len(suit_level)):   # iterate through each suitability level to project the coviriate values
                        # extract the criterion of crop attribution value of corresponding to each suitability level
                        value = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.item()]
                        if order == 'ascending':
                            # below are the logical rule of projecting crop attribution value to suitability level
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
            # the situation of both ascending and descending attribututes exist (Only for the continual attributes)
            # BE CAUTION!!! If it is a categorical coviriate and has only two crop attributes then the below script cannot deal with it
            # Currently assume the categorical coviriate must have 3 or more crop attributes
            elif len(crop_a) == 2:   
                value = [None] * 2
                order = [None] * 2
                value0 = [None] * 2
                crop_attr_value = [None] * 2
                for i in range(0, len(crop_a)):    
                    crop_attr_value[i] = list(subdf_crop[crop_a.iloc[i]])
                    order[i] = DetermineValueOrder(crop_attr_value[i])
                    
                if (order[0] is not None) and (order[1] is not None):
                    
                    # switch the order if necessary to make sure the fisrt crop attribute is always ascending 
                    if (order[0] == 'ascending') and (order[1] == 'descending'):
                        pass
                    elif (order[0] == 'descending') and (order[1] == 'ascending'):          
                        tmp = crop_a.iloc[0]
                        crop_a.iloc[0] = crop_a.iloc[1]
                        crop_a.iloc[1] = tmp
                        tmp = None
                    
                    # iterate through each suitability level    
                    for i in range(0, len(suit_level)):
                        # value[0] is for the ascending attribute while value[1] is for the descending attribute 
                        value[0] = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.iloc[0]]
                        value[1] = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.iloc[1]]
                        
                        # below are the logical rule of projecting crop attribution value to suitability level
                        if i == 0: 
                            if value[0] < value[1]:
                                print('Irrational values occur in {} or in {}. \nCannot generate suitability map of "{}".'.format(crop_a.iloc[0], crop_a.iloc[1], d_a))
                                break
                            suit_array = np.where(np.logical_and(property_array<=value[0], property_array>value[1]), suit_level[i], suit_array)
                        else:
                            # value0[0] is for the previous ascending attribute 
                            # value0[1] is for the previous descending attribute 
                            value0[0] = subdf_crop[subdf_crop['Suitability'] == suit_level[i-1]].iloc[0][crop_a.iloc[0]]
                            value0[1] = subdf_crop[subdf_crop['Suitability'] == suit_level[i-1]].iloc[0][crop_a.iloc[1]]
                            if ((np.isnan(value[0]) == False) and (value[0] != -9990)) and ((np.isnan(value[1]) == False) and (value[1] != -9999)):
                                suit_array = np.where(np.logical_or(np.logical_and(property_array>value0[0], property_array<=value[0]), np.logical_and(property_array<=value0[1], property_array>value[1])), suit_level[i], suit_array)
                            elif ((np.isnan(value[0]) == True) or (value[0] == -9990)) and ((np.isnan(value[1]) == False) and (value[1] != -9999)):
                                suit_array = np.where(np.logical_and(property_array<=value0[1], property_array>value[1]), suit_level[i], suit_array)                       
                                if (np.isnan(value0[0]) == False) and (value0[0] != -9990):
                                    suit_array = np.where(property_array>value0[0], suit_level[i], suit_array)
                            
                            elif ((np.isnan(value[0]) == False) and (value[0] != -9990)) and ((np.isnan(value[1]) == True) or (value[1] == -9999)):
                                suit_array = np.where(np.logical_and(property_array>value0[0], property_array<=value[0]), suit_level[i], suit_array)
                                if (np.isnan(value0[1]) == False) and (value0[1] != -9999):
                                    suit_array = np.where(property_array<=value0[1], suit_level[i], suit_array)
                                
                            else:
                                if ((np.isnan(value0[0]) == True) or (value0[0] == -9990)) and ((np.isnan(value0[1]) == False) and (value0[1] != -9999)):
                                    suit_array = np.where(property_array<=value0[1], suit_level[i], suit_array)
                                elif ((np.isnan(value0[0]) == False) and (value0[0] != -9990)) and ((np.isnan(value0[1]) == True) or (value0[1] == -9999)):
                                    suit_array = np.where(property_array>value0[0], suit_level[i], suit_array)
                                elif ((np.isnan(value0[0]) == False) and (value0[0] != -9990)) and ((np.isnan(value0[1]) == False) and (value0[1] != -9999)):
                                    suit_array = np.where(np.logical_or(property_array>value0[0], property_array<=value0[1]), suit_level[i], suit_array)

            # deal with the categorical coviriates
            # Again!!! assume categorical coviriates have 3 or more crop attributes (caterogries)
            elif len(crop_a) > 2:
                for j in range(0, len(crop_a)):    
                        
                    for i in range(0, len(suit_level)):
                        value = subdf_crop[subdf_crop['Suitability'] == suit_level[i]].iloc[0][crop_a.iloc[j]]
                        if np.isnan(value) == False:
                            suit_array = np.where(property_array==value, suit_level[i], suit_array)
                            break
            
            suit_array[np.where(property_array == NoData_value)] = NoData_value
            Array2Raster(suit_array, r, suit_raster_file)      # export the suitability map of each coviriate      

            suit_arrays.append(suit_array)
        
        crop_suit_array = ExtractMaxValueOfStack(suit_arrays)
        ref_array = homogenize_nodata_area(suit_arrays, NoData_value)
        crop_suit_array[np.where(ref_array == NoData_value)] = NoData_value
        crop_suit_raster_file = join(crop_folder, '{}_suitability.tif'.format(c))
        Array2Raster(crop_suit_array, suit_raster_file, crop_suit_raster_file)
        
        print('create dominant worst covariate raster at {}...'.format(dt.datetime.now()))
        worst_dominant_array, worst_count_array, worst_dominant_legend_list = ExtractMaxValueIndexBinaryOfStack(suit_arrays, covariate_list, 4)
        
        worst_dominant_array[np.where(ref_array == NoData_value)] = NoData_value
        worst_dominant_raster_file = join(crop_folder, '{}_worst_dominant.tif'.format(c))
        Array2Raster(worst_dominant_array, suit_raster_file, worst_dominant_raster_file)
        
        worst_count_array[np.where(ref_array == NoData_value)] = NoData_value
        worst_count_raster_file = join(crop_folder, '{}_worst_count.tif'.format(c))
        Array2Raster(worst_count_array, suit_raster_file, worst_count_raster_file)
        
        worst_dominant_legend_csv = join(crop_folder, '{}_worst_dominant_legend.csv'.format(c))
        csvw = CSVOperation.CSVWriting()
        headers = ['raster value', 'number of resriction', 'covariates']
        csvw.WriteLines(worst_dominant_legend_csv, headers, worst_dominant_legend_list)
        
# the commented part below is to deal with the multiple years iteration            
# =============================================================================
#             # differenciate the climatic and non-climatic coviriate array
#             if 'annual' in d_a: 
#                 climatic_arrays.append(suit_array)
#             else:
#                 non_climatic_arrays.append(suit_array)
#         
#         to_be_stacked_arrays = []   # the array list used to calculate the overall suitability
#         i = 0
#         for sa in climatic_arrays:    # iterate through each climate array and combine it to the non-climatic list
# 
#             to_be_stacked_arrays = []
#             to_be_stacked_arrays.extend(non_climatic_arrays)
#             to_be_stacked_arrays.append(sa)
#             
#             # the overall suitability array based on the maximum value of array list
#             crop_suit_array = ExtractMaxValueOfStack(to_be_stacked_arrays)
#             ref_array = homogenize_nodata_area(to_be_stacked_arrays, NoData_value)
#             crop_suit_array[np.where(ref_array == NoData_value)] = NoData_value
#             crop_suit_raster_file = join(crop_folder, '{}_suitability_{}.tif'.format(c, years[i]))
#             Array2Raster(crop_suit_array, suit_raster_file, crop_suit_raster_file)
#             i += 1
# =============================================================================

if __name__ == '__main__':
    
    config_path = r'D:\LandSuitability_AG\data\config'     # the path of the two configration files
    output_path = r'D:\LandSuitability_AG\output'          # the path of output rasters 
    data_path = join(output_path, 'original_property')     # the path of coviriate layers 
    
    if not os.path.exists(data_path):
        sys.exit('The directory of cofiguration files "{}" does not exist.'.format(data_path))
    
    # create a directory for suitability maps
    outSubRoot = join(output_path, 'suitability_maps2')
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
    print('Finished at {}.'.format(dt.datetime.now()))
        
        
        
        
        
        
        
    