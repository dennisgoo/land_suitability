# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:21:31 2018

This script is for SLMACC project. It creates climatic covariates including:
    1. Frost risk
    2. Growing degree days
    3. Chilling hours

@author: J. Guo
"""

from osgeo import gdal, osr
import numpy as np
import datetime as dt
import os.path
from os.path import join
#import sys
#import math
#import shutil
from os import walk
import configparser
import codecs


class Raster(object):
    '''
    Read raster file and return raster related information, e.g. raster array, nodata value etc..
    '''
    def __init__(self):
        pass

        
    def getRasterArray(self, file):
        try:
            r=gdal.Open(file)
        except:
            SystemExit("No such {} file exists!!!".format(file))
    
        self.band=r.GetRasterBand(1)
        rasterArray = self.band.ReadAsArray().astype(np.float)
        r = None
        return rasterArray
    
    def getNoDataValue(self, file):
        
        try:
            r=gdal.Open(file)
        except:
            SystemExit("No such {} file exists!!!".format(file))
    
        self.band=r.GetRasterBand(1)
        noDataValue = self.band.GetNoDataValue()
        r = None
        return noDataValue
    
    def array2Raster(self, in_array, ref_raster, new_raster):
        
        nodata_value = self.getNoDataValue(ref_raster)
        raster = gdal.Open(ref_raster)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        cols = in_array.shape[1]
        rows = in_array.shape[0]
    
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(new_raster, cols, rows, 1, gdal.GDT_Float32)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outband = outRaster.GetRasterBand(1)
        outband.SetNoDataValue(nodata_value)
        outband.WriteArray(in_array)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()


class ExtractTimeInfo(object):
    '''
    Extract an unique 'Year' list from a directory, and extract an unique 'Date' list based on input year. 
    '''
    def __init__(self, path):
        self.path = path
        self.file = []

        for (dirpath, dirnames, filenames) in walk(self.path):
            self.file.extend(filenames)
            break
        
    def extractYears(self):
        years = sorted(list(set([y.split('-')[0] for y in self.file])))
        return years
    
    def extractMonths(self, year):
        months = sorted(list(set([m.split('-')[1] for m in self.file if m.split('-')[0] == year])))
        return months
    
    def extractDates(self, year, month=None): 
        if month is None:
            dates = sorted(list(d.split('.')[0] for d in self.file if d.split('-')[0] == year and d.split('.')[-1] == 'tif'))
        else:
            dates = sorted(list(d.split('.')[0] for d in self.file if d.split('-')[0] == year and d.split('-')[1] == month and d.split('.')[-1] == 'tif'))
        return dates
    
class ClimaticCovariates(object):
    '''
    Climatic covariates class
    '''
    def __init__(self, year_list, data_dir):
        '''
        year_list: a numeric list of years which are took into account when calculating
                   climatic covariates
        '''
        self.years = year_list
        self.dir = data_dir
        self.ref_raster = ''

    def __GetFileList(self, start_date, end_date, keyword, tmax_kay = None):
        '''
        Return a file list for the given period of the years.
        'keyword' is the key to look for the correct files. 
        Based on the file/folder structure of Climate data, 
        here assumes one of the subfolder should contain the keyword. 
        '''
        files = []
        for (subdirpath, subdirname, filenames) in walk(self.dir):
            if keyword in subdirpath.split('\\')[-1]:    
                for f in filenames:    
                    d = f.split('.')[0].split('-')
                    if int(start_date[:2]) > int(end_date[:2]): 
                        for y in self.years[:-1]:
                            if (
                                    (
                                        (d[0] == y) and 
                                        (
                                            (d[1] == start_date[:2] and int(d[-1]) >= int(start_date[-2:])) or 
                                            (int(d[1]) > int(start_date[:2])) 
                                        )
                                    ) 
                                    or
                                    (
                                        (int(d[0]) == int(y)+1) and
                                        (
                                            (d[1] == end_date[:2] and int(d[-1]) <= int(end_date[-2:])) or 
                                            (int(d[1]) < int(end_date[:2]))            
                                        )
                                    )
                                ):
                                
                                files.append(join(subdirpath, f))
                    else:
                        for y in self.years:
                            if (
                                    (d[0] == y) and 
                                    (
                                        (d[1] == start_date[:2] and int(d[-1]) >= int(start_date[-2:])) or 
                                        (int(d[1]) > int(start_date[:2]) and int(d[1]) < int(end_date[:2])) or
                                        (d[1] == end_date[:2] and int(d[-1]) <= int(end_date[-2:]))
                                    )
                                ):
                                
                                files.append(join(subdirpath, f))
                        
        
        self.ref_raster = files[0]    
        
        return files
    
    
    def __ChillHoursModel__(self, tmin_array, tmax_array, base_min, base_max):

        tmin_array = np.where(tmin_array > tmax_array, tmax_array, tmin_array)
#        print('min: {}, max: {}'.format(tmin_array[43, 180], tmax_array[43, 180]))
        tave_array = (tmin_array + tmax_array) / 2
        
        denominator_array = tave_array - tmin_array
        daychill_array_A = np.where(denominator_array == 0, 0, np.where(tmax_array > base_max, 2 * 6 * (base_max - tmin_array) / (tave_array - tmin_array), 24))
        
#        daychill_array_A =  np.where(tmax_array > base_max, 2 * 6 * (base_max - tmin_array) / (tave_array - tmin_array), 24)
        daychill_array_B = np.where(denominator_array == 0, 0, np.where(tmin_array < base_min, 2 * 6 * (base_min - tmin_array) / (tave_array - tmin_array), 0))
#        daychill_array_B = np.where(tmin_array < base_min, 2 * 6 *(base_min - tmin_array) / (tave_array - tmin_array), 0)
        
        daychill_array = daychill_array_A - daychill_array_B
        daychill_array = np.where(daychill_array > 0, daychill_array, 0)

#        print(daychill_array_A[43, 180], daychill_array_B[43, 180])
        
        return daychill_array
    
    
    def GetFrostRiskArray(self, tmin_key, threshold_tmp, start_date, end_date):
        '''
        Frost risk frequency is determined by counting years 
        that had at least 1 day of frost occuring at less than
        the threshold temperature between a certain period in
        a year. This count was summed and divided by the total 
        number years.
        
        tmin_key:      key words of minimum temperature climate data subdir name (or filename)
        threshold_tmp: the temperature threshold of selected crop to determine
                       the occurance of frost risk (compared with tmin)
        start_date:    the start date of a certain period when frost risk is matter
        end_date:      the end date of a certain period when frost risk is matter
        
        '''

        file_list = self.__GetFileList(start_date, end_date, tmin_key)  # all the files to calculate the frost frequency
        raster = Raster()
        ref_array = raster.getRasterArray(self.ref_raster)  # a referency array from original raster to make sure the output raster has the same nodata area as it
        no_data = raster.getNoDataValue(self.ref_raster)
        frost_frequency_array = np.zeros(ref_array.shape)   # frosy frequency array
        
        # loop in each each year
        for year in self.years:
            daily_frost_array = np.zeros(ref_array.shape)
            accumu_daily_frost_array = np.zeros(ref_array.shape)
            for f in file_list:
                y, m, d = f.split('\\')[-1].split('.')[0].split('-')  #get the year, month and day of the file
                if int(end_date[:2]) >= int(start_date[:2]):    # if the certian period is within a natural year  
                    total_years = len(self.years)
                    if y == year:    # take all the files from the file list in that year
                        raster_array = raster.getRasterArray(f)
                        daily_frost_array = np.where(raster_array < threshold_tmp, 1, 0)                        
                        accumu_daily_frost_array = accumu_daily_frost_array + daily_frost_array
                else:   # if the certain period cross two natural years
                    total_years = len(self.years) - 1
                    if year == self.years[-1]: break   #when the certain period cross two years the last year in the list doesn't count
                    if (
                            (y == year and int(m) >= int(start_date[:2])) # take files in that year, and in or after the start month 
                             or 
                            (int(y) == int(year)+1 and int(m) <= int(end_date[:2])) # or take files in the next year, and in or before the end month 
                            
                        ):  
                        raster_array = raster.getRasterArray(f)
                        daily_frost_array = np.where(raster_array < threshold_tmp, 1, 0)  # when temperatrue below the base temperature frost happens (set the pixel value to 1)                      
                        accumu_daily_frost_array = accumu_daily_frost_array + daily_frost_array # accumulate the frost days over year
                    
            annual_frost_array = np.where(accumu_daily_frost_array != 0, 1, 0) # if not even 1 day frost happened set it to non frost year (pixel value=0) otherwise set it to frost year (pixel value=1) 
            frost_frequency_array = frost_frequency_array + annual_frost_array # accumulate the annual frost frequency
        
        frost_frequency_array = frost_frequency_array / total_years #get the average annual frost frequency
        frost_frequency_array = np.where(ref_array == no_data, no_data, frost_frequency_array)
        
        return frost_frequency_array
    
    def GetGDDArray(self, tmin_key, tmax_key, threshold_tmp, start_date, end_date):
        '''
        Growing Degree Days (GDD) is quantified for each day to give a GDD unit and
        is calculated by taking the average of the daily maximum and minmum temperatures
        compared to a base temperature. This count was summed and divided by the total 
        number years.
        
        tmin_key:      key words of minimum temperature climate data subdir name (or filename)
        tmax_key:      key words of maximum temperature climate data subdir name (or filename)
        threshold_tmp: the base temperature of selected crop to quantify GDD
        start_date:    the start date of a certain period when GDD is matter
        end_date:      the end date of a certain period when GDD is matter
        
        '''

        min_tmp_file_list = self.__GetFileList(start_date, end_date, tmin_key)
        max_tmp_file_list = self.__GetFileList(start_date, end_date, tmax_key)
        raster = Raster()
        ref_array = raster.getRasterArray(self.ref_raster)
        no_data = raster.getNoDataValue(self.ref_raster)
        GDD_array = np.zeros(ref_array.shape)
        
        for year in self.years:
            daily_GDD_array = np.zeros(ref_array.shape)
            accumu_daily_GDD_array = np.zeros(ref_array.shape)
            for minf, maxf in zip(min_tmp_file_list, max_tmp_file_list):
                if minf.split('\\')[-1] == maxf.split('\\')[-1]: # only excute when then min and max temperature files are in the same day (same filename)
                    y, m, d = minf.split('\\')[-1].split('.')[0].split('-')  #get the year, month and day of the file
                    if int(end_date[:2]) >= int(start_date[:2]): # if the certian period is within a natural year  
                        total_years = len(self.years)
                        if y == year:    # take all the files from the file list in that year
                            min_tmp_raster_array = raster.getRasterArray(minf)
                            max_tmp_raster_array = raster.getRasterArray(maxf)
                            daily_GDD_array = (min_tmp_raster_array + max_tmp_raster_array) / 2 - threshold_tmp
                            accumu_daily_GDD_array = accumu_daily_GDD_array + daily_GDD_array
                    
                    else: # if the certain period cross two natural years
                        total_years = len(self.years) - 1
                        if year == self.years[-1]: break   #when the certain period cross two years the last year in the list doesn't count
                        if (
                                (y == year and int(m) >= int(start_date[:2])) # take files in that year, and in or after the start month 
                                 or 
                                (int(y) == int(year)+1 and int(m) <= int(end_date[:2])) # or take files in the next year, and in or before the end month 
                                
                            ):
                            min_tmp_raster_array = raster.getRasterArray(minf)
                            max_tmp_raster_array = raster.getRasterArray(maxf)
                            daily_GDD_array = (min_tmp_raster_array + max_tmp_raster_array) / 2 - threshold_tmp
                            daily_GDD_array = np.where(daily_GDD_array<=0, 0, daily_GDD_array)
#                            print('{}-{}-{}\nmin: {}, max: {}, GDD: {}'.format(y,m,d,min_tmp_raster_array[140, 137], max_tmp_raster_array[140, 137], daily_GDD_array[140, 137]))
                            accumu_daily_GDD_array = accumu_daily_GDD_array + daily_GDD_array
                    
            GDD_array = GDD_array + accumu_daily_GDD_array 
        
        GDD_array = GDD_array / total_years
        GDD_array = np.where(ref_array == no_data, no_data, GDD_array)
        
        return GDD_array
    
    def GetChillHoursArray(self, tmin_key, tmax_key, threshold_tmp_min, threshold_tmp_max, start_date, end_date):
        '''
        Chill hours are calculated as the number of hours in 
        a temperature range of (threshold_tmp_min to threshold_tmp_max).
        This count was summed and divided by the total number years.
        
        *Note: As we don't have hourly temperature data, we use to daily data to model chill hours.
               The model is provided by Anne-Gaelle Ausseil (in Winterchillhours_HB.rmd) 
        
        tmin_key:           key words of minimum temperature climate data subdir name (or filename)
        tmax_key:           key words of maximum temperature climate data subdir name (or filename)
        threshold_tmp_min:  the min temperature threshold of selected crop to calculate chill hours
        threshold_tmp_max:  the max temperature threshold of selected crop to calculate chill hours
        start_date:         the start date of a certain period when frost risk is matter
        end_date:           the end date of a certain period when frost risk is matter
        
        '''

        min_tmp_file_list = self.__GetFileList(start_date, end_date, tmin_key)
        max_tmp_file_list = self.__GetFileList(start_date, end_date, tmax_key)
        raster = Raster()
        ref_array = raster.getRasterArray(self.ref_raster)
        no_data = raster.getNoDataValue(self.ref_raster)
        chill_hours_array = np.zeros(ref_array.shape)
        
        for year in self.years:
            daily_chill_array = np.zeros(ref_array.shape)
            accumu_daily_chill_array = np.zeros(ref_array.shape)
            for minf, maxf in zip(min_tmp_file_list, max_tmp_file_list):
                if minf.split('\\')[-1] == maxf.split('\\')[-1]: # only excute when then min and max temperature files are in the same day (same filename)
                    y, m, d = minf.split('\\')[-1].split('.')[0].split('-')  #get the year, month and day of the file
                    if int(end_date[:2]) >= int(start_date[:2]): # if the certian period is within a natural year  
                        total_years = len(self.years)
                        if y == year:    # take all the files from the file list in that year
                            min_tmp_raster_array = raster.getRasterArray(minf)
                            max_tmp_raster_array = raster.getRasterArray(maxf)
                            daily_chill_array = self.__ChillHoursModel__(min_tmp_raster_array, max_tmp_raster_array, threshold_tmp_min, threshold_tmp_max)
                            accumu_daily_chill_array = accumu_daily_chill_array + daily_chill_array
#                            print('{}-{}-{}\nmin: {}, max: {}, chill: {}'.format(y,m,d,min_tmp_raster_array[43, 180], max_tmp_raster_array[43, 180], daily_chill_array[43, 180]))
                    else: # if the certain period cross two natural years
                        total_years = len(self.years) - 1
                        if year == self.years[-1]: break   #when the certain period cross two years the last year in the list doesn't count
                        if (
                                (y == year and int(m) >= int(start_date[:2])) # take files in that year, and in or after the start month 
                                 or 
                                (int(y) == int(year)+1 and int(m) <= int(end_date[:2])) # or take files in the next year, and in or before the end month 
                                
                            ):
                            min_tmp_raster_array = raster.getRasterArray(minf)
                            max_tmp_raster_array = raster.getRasterArray(maxf)
                            print('al-{}-{}-{}'.format(y,m,d))
                            daily_chill_array = self.__ChillHoursModel__(min_tmp_raster_array, max_tmp_raster_array, threshold_tmp_min, threshold_tmp_max)
                            accumu_daily_chill_array = accumu_daily_chill_array + daily_chill_array
                            print(np.max(accumu_daily_chill_array))
                            
            chill_hours_array = chill_hours_array + accumu_daily_chill_array 
        
        chill_hours_array = chill_hours_array / total_years
        chill_hours_array = np.where(ref_array == no_data, no_data, chill_hours_array)
        
        return chill_hours_array
    

def headerDictionary():
    param_header = {
                        'crop': 'crop',
                        'f_s_d': 'frost_start_date',
                        'f_e_d': 'frost_end_date',
                        'f_b_t': 'frost_base_temp',
                        'f_h_s_d': 'frost_harvest_start_date',
                        'f_h_e_d': 'frost_harvest_end_date',
                        'f_h_b_t': 'frost_harvest_base_temp',
                        'g_s_d':  'gdd_start_date',
                        'g_e_d': 'gdd_end_date',
                        'g_b_t': 'gdd_base_temp',
                        'c_s_d': 'chill_start_date',
                        'c_e_d': 'chill_end_date',
                        'c_bmin_t': 'chill_base_min_temp',
                        'c_bmax_t': 'chill_base_max_temp',
                        'key_min': 'key_min_temperature',
                        'key_max': 'key_max_temperature'
                    }
    return param_header

def getYearList(start_year, end_year):
    year_list = []
    y = start_year
    while y <= end_year:
        year_list.append(str(y))
        y += 1
    return year_list

if __name__ == '__main__':
    
    conf = input('Please enter the full dir and filename of config file\n(Enter): ')
    
    while not os.path.isfile(conf):
        print('The config.ini file does not exist. Would you like to use the default file?')
        is_default = input('(Yes/No): ')
        if is_default[0].lower() == 'y': 
            conf = r'D:\LandSuitability_AG\data\config\config.ini'
        else:
            conf = input('Please enter the full dir and filename of config file.\nOr leave it blanket to point to the default file\n (Enter): ')
    
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read_file(codecs.open(conf, 'r', 'utf8'))
    
    proj_header = 'projectConfig'
    num_crops = int(config.get(proj_header,'num_crops'))
    climate_data_dir = config.get(proj_header,'climateSource')
    out_dir = config.get(proj_header,'outDir')
    
    s_y = int(config.get(proj_header,'start_year'))
    e_y = int(config.get(proj_header,'end_year'))
    year_list = getYearList(s_y, e_y)
    
    covariate_dir = join(out_dir, 'climatic_covariates')
    if not os.path.exists(covariate_dir):
        os.makedirs(covariate_dir, exist_ok = True)
        
    climate_covariate = ClimaticCovariates(year_list, climate_data_dir)
    
    param_header = headerDictionary()
    
    out_raster = Raster()
    for i in range(1, num_crops+1):
        crop_header = 'crop_{}'.format(i)
        
        crop = config.get(crop_header,param_header['crop'])
        key_min = config.get(crop_header,param_header['key_min'])
        key_max = config.get(crop_header,param_header['key_max'])
        
# =============================================================================
#         # Create Frost frequency raster
#         f_s_d = config.get(crop_header,param_header['f_s_d'])
#         f_e_d = config.get(crop_header,param_header['f_e_d'])
#         f_b_t = float(config.get(crop_header,param_header['f_b_t'])) + 273.15
#         
#         frost_frequency_array = climate_covariate.GetFrostRiskArray(key_min, f_b_t, f_s_d, f_e_d)        
#         ref_raster = climate_covariate.ref_raster
#         frost_raster = join(covariate_dir, '{}_frost_{}_{}.tif'.format(crop, s_y, e_y))
#         out_raster.array2Raster(frost_frequency_array, ref_raster, frost_raster)
#         
#         #Frost After fruit set - harvest
#         f_s_d = config.get(crop_header,param_header['f_h_s_d'])
#         f_e_d = config.get(crop_header,param_header['f_h_e_d'])
#         f_b_t = float(config.get(crop_header,param_header['f_h_b_t'])) + 273.15
#         frost_frequency_array = climate_covariate.GetFrostRiskArray(key_min, f_b_t, f_s_d, f_e_d)
#         frost_raster = join(covariate_dir, '{}_frost_harvest_{}_{}.tif'.format(crop, s_y, e_y))
#         out_raster.array2Raster(frost_frequency_array, ref_raster, frost_raster)
#         
#         # Create GDD raster
#         g_s_d = config.get(crop_header,param_header['g_s_d'])
#         g_e_d = config.get(crop_header,param_header['g_e_d'])
#         g_b_t = float(config.get(crop_header,param_header['g_b_t'])) + 273.15
#         
#         GDD_array = climate_covariate.GetGDDArray(key_min, key_max, g_b_t, g_s_d, g_e_d)
#         
#         GDD_raster = join(covariate_dir, '{}_GDD_{}_{}.tif'.format(crop, s_y, e_y))
#         out_raster.array2Raster(GDD_array, ref_raster, GDD_raster)
# =============================================================================
        
        # Create chill hours raster
        c_s_d = config.get(crop_header,param_header['c_s_d'])
        c_e_d = config.get(crop_header,param_header['c_e_d'])
        c_bmin_t = float(config.get(crop_header,param_header['c_bmin_t'])) + 273.15
        c_bmax_t = float(config.get(crop_header,param_header['c_bmax_t'])) + 273.15
        
        chill_array = climate_covariate.GetChillHoursArray(key_min, key_max, c_bmin_t, c_bmax_t, c_s_d, c_e_d)
        ref_raster = climate_covariate.ref_raster
        chill_raster = join(covariate_dir, '{}_chill_{}_{}.tif'.format(crop, s_y, e_y))
        out_raster.array2Raster(chill_array, ref_raster, chill_raster)
        
        
        