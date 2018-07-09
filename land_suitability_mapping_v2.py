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
Revised on 29 June 2018 (v2)
Land suitability model
@author: Jing Guo
"""

import os
from os.path import join

import numpy as np
import sys
from os import walk
import CSVOperation
import datetime as dt

from config import ConfigParameters
from raster import Raster
from sqlite_conn import Sqlite_connection

# =============================================================================
# def DataTypeConversion(x):
#     return {
#             'Real': gdal.GDT_Float32,
#             'Integer': gdal.GDT_Int32,
#         }.get(x, gdal.GDT_Unknown)
#     
# 
# def DataTypeConversion_GDAL2NP(x):
#     return {
#             5: np.int,
#             6: np.float,
#         }.get(x, np.float)
# =============================================================================

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

def strip_start(text, suffix):
    if not text.startswith(suffix):
        return text
    return text[len(suffix):]


class LandSuitability(object):
    
    def __init__(self):
        self.no_data = ''
    
    def __reclassify_contianual__(self, covariate_array, rulesets):
        
        suit_array = np.zeros(covariate_array.shape)  # an empty array for suitability array (fill with value afterwards)
        direction = ''
        
        for row in rulesets:
            suit_level = row['suitability_level']
            low1 = row['low_value']
            high1 = row['high_value']
            low2 = row['low_value_2']
            high2 = row['high_value_2']
            
            # Here we deal with the suitability level one, from which we derive the direction of covariate response curve
            if suit_level == 1:
                # This is bell shaped curve situation 
                # (only under this circumstance low2 or high2 of the rest of suitability level may exist)
                if low1 is not None and high1 is not None: 
                    direction = 'two'
                    suit_array = np.where(np.logical_and(covariate_array>=low1, covariate_array<high1), suit_level, suit_array)
                                  
                # This is one direction descending situation
                elif low1 is not None and high1 is None:     
                    direction = 'descd'
                    suit_array = np.where(covariate_array>=low1, suit_level, suit_array)
                    
                # This is one direction ascending situation
                elif low1 is None and high1 is not None:     
                    direction = 'ascd'
                    suit_array = np.where(covariate_array<=high1, suit_level, suit_array)
                    
                # This is low1 and high1 are both None, which shouldn't exist    
                else:
                    print('Warning! Unexpected "None" value exist in both "low_value" and "high_value" of' 
                          'suitability level 1 of crop {}, covariate {}. Please check the database and' 
                          'set the correct values.'.format(row['crop_id'], row['covariate_id']))
            
            # From here we dealing with the rest of suitability levels
            else:
                # If it is one direction (ascending or descending), then we don't have to care about the low2 and high2
                if direction == 'descd':
                    if low1 is not None and high1 is not None:
                        suit_array = np.where(np.logical_and(covariate_array>=low1, covariate_array<high1), suit_level, suit_array)
                    elif low1 is None and high1 is not None:
                        suit_array = np.where(covariate_array<high1, suit_level, suit_array)

                elif direction == 'ascd':
                    if low1 is not None and high1 is not None:
                        suit_array = np.where(np.logical_and(covariate_array>low1, covariate_array<=high1), suit_level, suit_array)
                    elif low1 is not None and high1 is None:
                        suit_array = np.where(covariate_array>low1, suit_level, suit_array)
                
                # If it is two direction then we deal with the most complex situation 
                else:
                    if low1 is not None and high1 is not None:
                        if low2 is not None and high2 is not None:
                            suit_array = np.where(np.logical_or(np.logical_and(covariate_array>=low1, covariate_array<high1), 
                                                                np.logical_and(covariate_array>=low2, covariate_array<high2)), 
                                                  suit_level, suit_array)
                        elif low2 is None and high2 is not None:
                            suit_array = np.where(np.logical_or(np.logical_and(covariate_array>=low1, covariate_array<high1), 
                                                                covariate_array<high2), 
                                                  suit_level, suit_array)
                        elif low2 is not None and high2 is None:
                            suit_array = np.where(np.logical_or(np.logical_and(covariate_array>=low1, covariate_array<high1), 
                                                                covariate_array>=low2), 
                                                  suit_level, suit_array)
                        else:
                            suit_array = np.where(np.logical_and(covariate_array>=low1, covariate_array<high1), suit_level, suit_array)
                        
                    elif low1 is None and high1 is not None:
                        if low2 is not None and high2 is not None:
                            suit_array = np.where(np.logical_or(covariate_array<high1, 
                                                                np.logical_and(covariate_array>=low2, covariate_array<high2)), 
                                                  suit_level, suit_array)
                        elif low2 is not None and high2 is None:
                            suit_array = np.where(np.logical_or(covariate_array<high1, 
                                                                covariate_array>=low2), 
                                                  suit_level, suit_array)
                        #This situation shouln't exist, because bath low1 and low2 are none means tow parts are in the same direction
                        elif low2 is None and high2 is not None:
                            print('Warning! Unexpected value exist in either "low_value_1" or "low_value_2" of suitability'
                                  'level {} of crop {}, covariate {}. Please check the database and set the correct'
                                  'values.'.format(row['suitability_level'], row['crop_id'], row['covariate_id']))
                        else:
                            suit_array = np.where(covariate_array<high1, suit_level, suit_array)
                    
                    elif low1 is not None and high1 is None:
                        if low2 is not None and high2 is not None:
                            suit_array = np.where(np.logical_or(covariate_array>=low1, 
                                                                np.logical_and(covariate_array>=low2, covariate_array<high2)), 
                                                  suit_level, suit_array)
                        elif low2 is None and high2 is not None:
                            suit_array = np.where(np.logical_or(covariate_array>=low1, 
                                                                covariate_array<high2), 
                                                  suit_level, suit_array)
                        #This situation shouln't exist, because bath low1 and low2 are none means tow parts are in the same direction
                        elif low2 is not None and high2 is None:
                            print('Warning! Unexpected value exist in either "high_value_1" or "high_value_2" of suitability'
                                  'level {} of crop {}, covariate {}. Please check the database and set the correct'
                                  'values.'.format(row['suitability_level'], row['crop_id'], row['covariate_id']))
                        else:
                            suit_array = np.where(covariate_array>low1, suit_level, suit_array)
                    
                    # When both low1 and high1 are None, low2 and high2 must be None, otherwise we should put the low2 and high2 to low1 and high1
                    else:
                        if low2 is not None or high2 is not None:
                            print('Warning! Unexpected value exist in either "low_value_2" or "high_value_2" of suitability'
                                  'level {} of crop {}, covariate {}. Please check the database and set the correct'
                                  'values.'.format(row['suitability_level'], row['crop_id'], row['covariate_id']))
                            
        return suit_array
    
    def __reclassify_catorgorical__(self, covariate_array, rulesets):            
                
        suit_array = np.zeros(covariate_array.shape)  # an empty array for suitability array (fill with value afterwards)
        
        all_cat_values_not_level4 = []
        for row in rulesets:
            suit_level = row['suitability_level']
            
            if row['cat_value'] is not None:
                cat_values = [int(x) for x in row['cat_value'].split(',')]
                suit_array = np.where(np.isin(covariate_array, cat_values), suit_level, suit_array)
                all_cat_values_not_level4 = all_cat_values_not_level4 + cat_values
            
            # If the catorgorical value is None at certian suitability level, then we check if it is at levle 4. 
            # If so then we set the rest of class values to level 4, otherwise don't create that level. 
            else:
                if suit_level == 4:
                    # Set all cells, where their class value is not in the all_cat_values_not_level4 list, to level 4 
                    suit_array = np.where(np.isin(covariate_array, all_cat_values_not_level4, invert=True), suit_level, suit_array)
        
        return suit_array
    
    def mapping(self, crop_id, covariate_rasters, conn, covariate_root_dir, suit_root_dir):
        
        raster = Raster()
        self.no_data =raster.getNoDataValue(covariate_rasters[-1])
        ref_rst = covariate_rasters[-1]
        covariate_id_list = []
        suit_array_stack = []
        
        filepath = strip_end(ref_rst, ref_rst.split('\\')[-1])
        out_dir = join(suit_root_dir, strip_start(filepath, covariate_root_dir)[1:]) # the [1:] is for removing the first and the last '\' of string
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok = True)
         
        for rst in covariate_rasters:
            
            filename = rst.split('\\')[-1].split('.')[0]
            
            if len(filename.split('_')) == 1:
                covariate_id = filename
                covariate_id_list.append(covariate_id)
                out_raster = join(out_dir, 'suitability_{}_{}.tif'.format(crop_id, covariate_id))
            else:
                covariate_id = filename.split('_')[1]
                covariate_id_list.append(covariate_id)
                time_span = '{}_{}'.format(filename.split('_')[-2], filename.split('_')[-1])
                out_raster = join(out_dir, 'suitability_{}_{}_{}.tif'.format(crop_id, covariate_id, time_span))
            
            with conn as cur:
                rows = cur.execute("select * from suitability_rule where crop_id=? and covariate_id=? order by suitability_level", (crop_id,covariate_id,)).fetchall()
                is_continual = cur.execute("select * from Covariate where id=?", (covariate_id,)).fetchone()['iscontinual']
            
            # If the query returns none then move to the next covariate
            if rows:
                covariate_array = raster.getRasterArray(rst)  # array of the covariate
                
                if is_continual == 1:
                    suit_array = self.__reclassify_contianual__(covariate_array, rows)
                else:
                    suit_array = self.__reclassify_catorgorical__(covariate_array, rows)
                
                suit_array[np.where(covariate_array == self.no_data)] = self.no_data
                
                raster.array2Raster(suit_array, ref_rst, out_raster)
                
                suit_array_stack.append(suit_array)
            else:
                print('Warning! Suitability ruleset for {}, {} not found! Please check the database.'.format(crop_id, covariate_id))
        
        if len(suit_array_stack) > 0:
            crop_suit_array = ExtractMaxValueOfStack(suit_array_stack)
            ref_array = homogenize_nodata_area(suit_array_stack, self.no_data)
            crop_suit_array[np.where(ref_array == self.no_data)] = self.no_data
            crop_suit_raster = join(out_dir, '{}_suitability.tif'.format(crop_id))
            raster.array2Raster(crop_suit_array, ref_rst, crop_suit_raster)
            
            print('create dominant worst covariate raster at {}...'.format(dt.datetime.now()))
            worst_dominant_array, worst_count_array, worst_dominant_legend_list = ExtractMaxValueIndexBinaryOfStack(suit_array_stack, covariate_id_list, 4)
            
            worst_dominant_array[np.where(ref_array == self.no_data)] = self.no_data
            worst_dominant_raster_file = join(out_dir, '{}_worst_dominant.tif'.format(crop_id))
            raster.array2Raster(worst_dominant_array, ref_rst, worst_dominant_raster_file)
            
            worst_count_array[np.where(ref_array == self.no_data)] = self.no_data
            worst_count_raster_file = join(out_dir, '{}_worst_count.tif'.format(crop_id))
            raster.array2Raster(worst_count_array, ref_rst, worst_count_raster_file)
            
            worst_dominant_legend_csv = join(out_dir, '{}_worst_dominant_legend.csv'.format(crop_id))
            csvw = CSVOperation.CSVWriting()
            headers = ['raster value', 'number of restriction', 'covariates']
            csvw.WriteLines(worst_dominant_legend_csv, headers, worst_dominant_legend_list)
        else:
            print('Warning! No suitability map for {} was created!'.format(crop_id))


def get_cova_raster_list(crop, root_dir):
    
    has_climate = False
    covariate_rasters = []
    for (dirpath, subdirname, filenames) in walk(root_dir):
        if dirpath == root_dir:  # get all the raster under root dir which are share used covariate raster such as slope, ph etc.. 
            for f in filenames:
                if f.split('.')[-1].lower()[:3] == 'tif':
                    covariate_rasters.append(join(dirpath, f))
                    
        if dirpath.split('\\')[-1] == crop:
            for f in filenames:
                if f.split('.')[-1].lower()[:3] == 'tif':
                    covariate_rasters.append(join(dirpath, f))
                    has_climate = True
            break
    return covariate_rasters, has_climate

    
def main():    
    
    conf = input('Please enter the full dir and filename of config file\n(Enter): ')
    
    while not os.path.isfile(conf):
        print('The config.ini file does not exist. Would you like to use the default file?')
        is_default = input('(Yes/No): ')
        if is_default[0].lower() == 'y': 
            conf = r'config.ini'
        else:
            conf = input('Please enter the full dir and filename of config file.\nOr leave it blank to point to the default file\n (Enter): ')
    
    config_params = ConfigParameters(conf)
    proj_header = 'projectConfig'
#    sui_header = 'landSuitability'
    
    db_file = config_params.GetDB(proj_header)
    covariates_dir = config_params.GetProcessedCovariateDir(proj_header)
    suit_map_dir = config_params.GetSuitabilityParams(proj_header)
    
    if not os.path.exists(covariates_dir):
        sys.exit('The directory of cofiguration files "{}" does not exist.'.format(covariates_dir))
    
    
    conn = Sqlite_connection(db_file)
    
    crops_id = []
    crops = []
    
    with conn as cur:
        rows = cur.execute("select * from crop").fetchall()
        
    for row in rows:
        crops_id.append(row['id'])
        crops.append(row['crop'])
    
    landsuit = LandSuitability()
    
    for crop_id, crop in zip(crops_id, crops):
        
        print('Processing {} at {}.'.format(crop, dt.datetime.now()))
        
        covariate_rasters, has_climate = get_cova_raster_list(crop, covariates_dir)
        
        if len(covariate_rasters) > 0 and has_climate == True:
            landsuit.mapping(crop_id, covariate_rasters, conn, covariates_dir, suit_map_dir)
        else:
            print('Warning! Covariate raster for {} not found!'.format(crop))
    
    print('End at {}.'.format(dt.datetime.now()))
    
    
if __name__ == '__main__':
    
    main()    
    
    
        
        
        
        
        
        
        
    