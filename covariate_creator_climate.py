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

class ExtractTimeInfo2(ExtractTimeInfo): 
    
    def __init__(self, file_list):
        self.file = file_list
    
class ClimaticCovariates(object):
    '''
    Climatic covariates class
    '''
    def __init__(self, year_list, data_dir):
        '''
        year_list: a numeric list of years which are took into account when calculating
                   climatic covariates
        '''  
        def GetRefRaster(data_dir):
            for (subdirpath, subdirname, filenames) in walk(data_dir):
                for f in filenames:
                    if f.split('.')[-1].lower()[:3] == 'tif':
                        return join(subdirpath, f)
        
        self.years = year_list
        self.dir = data_dir
        self.raster = Raster()
        self.ref_raster = GetRefRaster(self.dir)
        self.ref_array = self.raster.getRasterArray(self.ref_raster)
        self.no_data = self.raster.getNoDataValue(self.ref_raster)


    def __GetFileList__(self, start_date, end_date, keyword):
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
                    # get the date of each file
                    d = f.split('.')[0].split('-')
                    # if the month of the start date is later than that of the end date
                    # which means the given period is across two natural years
                    if int(start_date[:2]) > int(end_date[:2]): 
                        for y in self.years[:-1]:
                            if (        # in that case
                                    (   # get files of that year
                                        (d[0] == y) and 
                                        (   # and the date after the start date
                                            (d[1] == start_date[:2] and int(d[-1]) >= int(start_date[-2:])) or 
                                            (int(d[1]) > int(start_date[:2])) 
                                        )
                                    ) 
                                    or  
                                    (   # also get files of the next year
                                        (int(d[0]) == int(y)+1) and
                                        (   # and the date before the end date
                                            (d[1] == end_date[:2] and int(d[-1]) <= int(end_date[-2:])) or 
                                            (int(d[1]) < int(end_date[:2]))            
                                        )
                                    )
                                ):
                                
                                files.append(join(subdirpath, f))
                    else: # the given period in one natural year
                        for y in self.years:
                            if (    # get files of that year
                                    (d[0] == y) and 
                                    (   # and the date in between the start and the end date
                                        (d[1] == start_date[:2] and int(d[-1]) >= int(start_date[-2:])) or 
                                        (int(d[1]) > int(start_date[:2]) and int(d[1]) < int(end_date[:2])) or
                                        (d[1] == end_date[:2] and int(d[-1]) <= int(end_date[-2:]))
                                    )
                                ):
                                
                                files.append(join(subdirpath, f))
        
        return files
    
    
    def __GetFileDictionary__(self, start_date, end_date, keyword):
        '''
        From the file list generated from the __GetFileList__ function,
        return a file Dictionary with year named key and a file list related to the key.
        '''
        file_dict = {}
        file_list = self.__GetFileList__(start_date, end_date, keyword)

        if int(start_date[:2]) > int(end_date[:2]): 
            for year in self.years[:-1]:
                files = []
                for f in file_list:
                    y, m, d = f.split('\\')[-1].split('.')[0].split('-')
                    if (
                            (
                                (y == year) and 
                                (
                                    (m == start_date[:2] and int(d) >= int(start_date[-2:])) or 
                                    (int(m) > int(start_date[:2])) 
                                )
                            ) 
                            or
                            (
                                (int(y) == int(year)+1) and
                                (
                                    (m == end_date[:2] and int(d) <= int(end_date[-2:])) or 
                                    (int(m) < int(end_date[:2]))            
                                )
                            )
                        ):
                        
                        files.append(f)
                file_dict[year] = files
                
        else:
            for year in self.years:
                files = []
                for f in file_list:
                    y, m, d = f.split('\\')[-1].split('.')[0].split('-')
                    if (
                            (y == year) and 
                            (
                                (m == start_date[:2] and int(d) >= int(start_date[-2:])) or 
                                (int(m) > int(start_date[:2]) and int(m) < int(end_date[:2])) or
                                (m == end_date[:2] and int(d) <= int(end_date[-2:]))
                            )
                        ):
                        
                        files.append(f)
                file_dict[year] = files         
        
        return file_dict
    
    
    def __ChillHoursModel__(self, tmin_array, tmax_array, base_min, base_max):
        '''
        The model of simulating chill hours based on daily temperature data.
        
        *Note: 1. the original model has an issue on the denominator (tave_array - tmin_array) 
                  of the algorithm, when it is equel to 0. 
               2. some pixels of the daily temperature data have abnormal values e.g. 
                  the min is greater than the max (e.g. pixel [43, 180] from 1971-05-01).
               So in this function a reset (when abnormal values occur) of min temperature is 
               coded at the beginning, to eliminate the effect. But this may result in other 
               issues such as an unexpected result.
               
               3. When negtive chill hours occur set it to 0 (may not the correct way)
        '''
        tmin_array = np.where(tmin_array >= tmax_array, tmax_array - 1, tmin_array)
        tave_array = (tmin_array + tmax_array) / 2
        daychill_array_A =  np.where(tmax_array > base_max, 2 * 6 * (base_max - tmin_array) / (tave_array - tmin_array), 24)
        daychill_array_B = np.where(tmin_array < base_min, 2 * 6 *(base_min - tmin_array) / (tave_array - tmin_array), 0)
        
        daychill_array = daychill_array_A - daychill_array_B
        daychill_array = np.where(daychill_array > 0, daychill_array, 0)
        
        return daychill_array
        
    
    def __AnnualTemperatrueFrequency__(self, annual_file_list, base_temp, direction):
        
        '''
        Temperatrue Frequency includes frost risk frequency and max daily temperature etc. 
        It is determined by counting years that had at least 1 day of extreme temperature 
        occuring at less than the threshold temperature between a certain period in an 
        agriculture year. 
        This function returns a raster array of the frequency within one agriculture year, 
        but it can be called multiple times to get an average frequency of multiple years.
        
        annual_file_list: a file list contains the daily temperature file in a certain 
                          period when frost risk is matter in an agriculture year
        base_temp:        the temperature threshold of selected crop to determine
                          the occurance of frost risk (compared with the base temperature)
        direction:        a keyword which is either 'above' or 'below' to determine 
                          the temperature interval that exceeds the base temperature
        '''
        
        accumu_daily_array = np.zeros(self.ref_array.shape)
        for f in annual_file_list:
            raster_array = self.raster.getRasterArray(f)
            if direction == 'below':
                daily_array = np.where(raster_array < base_temp, 1, 0) 
            elif direction == 'above':
                daily_array = np.where(raster_array > base_temp, 1, 0) 
            accumu_daily_array = accumu_daily_array + daily_array
        
        annual_frequency_array = np.where(accumu_daily_array != 0, 1, 0)
        return annual_frequency_array
    
    def __AnnualGDD__(self, annual_file_list_min, annual_file_list_max, base_temp):
        
        '''
        Growing Degree Days (GDD) is quantified for each day to give a GDD unit and
        is calculated by taking the average of the daily maximum and minmum temperatures
        compared to a base temperature. This function returns a raster array of the GDD
        within one agriculture year, but it can be called multiple times to get an average 
        GDD of multiple years.
        
        annual_file_list_min: a file list contains the daily min temperature file in a certain 
                              period when GDD is matter in an agriculture year
        annual_file_list_max: a file list contains the daily max temperature file in a certain 
                              period when GDD is matter in an agriculture year
        base_temp:            the base temperature of selected crop to quantify GDD
        '''
        
        accumu_daily_array = np.zeros(self.ref_array.shape)
        for minf, maxf in zip(annual_file_list_min, annual_file_list_max):
            min_raster_array = self.raster.getRasterArray(minf)
            max_raster_array = self.raster.getRasterArray(maxf)
            daily_array = (min_raster_array + max_raster_array) / 2 - base_temp
            daily_array = np.where(daily_array<=0, 0, daily_array)                       
            accumu_daily_array = accumu_daily_array + daily_array
        
        return accumu_daily_array
    
    def __AnnualChillHours__(self, annual_file_list_min, annual_file_list_max, base_temp_min, base_temp_max):
        
        '''
        Chill hours are calculated as the number of hours in 
        a temperature range of (threshold_tmp_min to threshold_tmp_max).
        This count was summed and divided by the total number years.
        
        *Note: As the hourly temperature data is not available, we use to daily data to model 
               chill hours (with '__ChillHoursModel__' function).The model is provided by 
               Anne-Gaelle Ausseil (from Winterchillhours_HB.rmd) 
        
        annual_file_list_min: a file list contains the daily min temperature file in a certain 
                              period when chill hours is matter in an agriculture year
        annual_file_list_max: a file list contains the daily max temperature file in a certain 
                              period when chill hours is matter in an agriculture year
        base_temp_min:        the min base temperature threshold of selected crop to calculate 
                              chill hours
        base_temp_max:        the max base temperature threshold of selected crop to calculate 
                              chill hours
        '''
        
        accumu_daily_array = np.zeros(self.ref_array.shape)
        for minf, maxf in zip(annual_file_list_min, annual_file_list_max):
            min_raster_array = self.raster.getRasterArray(minf)
            max_raster_array = self.raster.getRasterArray(maxf)
            daily_array = self.__ChillHoursModel__(min_raster_array, max_raster_array, base_temp_min, base_temp_max)                        
            accumu_daily_array = accumu_daily_array + daily_array
        
        return accumu_daily_array
    
    def __AnnualMeanDaily__(self, annual_file_list):
        '''
        Annual mean daily temperature
        '''
        annual_average_array = np.zeros(self.ref_array.shape)
        for f in annual_file_list:
            raster_array = self.raster.getRasterArray(f)
            annual_average_array = annual_average_array + raster_array
        
        annual_average_array = annual_average_array / len(annual_file_list)
        annual_average_array = annual_average_array - 273.15
        return annual_average_array
    
    def __AnnualMeanMonthlyBasedOnDaily__(self, year, annual_file_list, method):
        
        '''
        Mean monthly temperature (can be min, max or average).
        First iterate each month to calculate the monthly temperature, 
        then get the annual mean
        '''
        
        annual_average_array = np.zeros(self.ref_array.shape)
        monthly_accumulated_array = np.zeros(self.ref_array.shape)
        time_info = ExtractTimeInfo2(annual_file_list)
        months = time_info.extractMonths(year)
        
        for m in months:
            i = 0
            for f in annual_file_list:
                if f.split('\\')[-1].split('-')[1] == m:
                    raster_array = self.raster.getRasterArray(f)
                    if i == 0:
                        target_array = raster_array
                    else:
                        if method.lower()[:3] == 'max':
                            target_array = np.maximum(target_array, raster_array)
                        elif method.lower()[:3]  == 'min':
                            target_array = np.minimum(target_array, raster_array)
                        elif method.lower()[:3]  == 'ave':
                            target_array = np.add(target_array, raster_array)
                        else:
                            print('Warning! No correct array aggregation method was passed to the function "__AnnualMeanMonthlyBasedOnDaily__()". The calculation will the the Max vaule of each array!')
                            target_array = np.maximum(target_array, raster_array)
                    i += 1
            
            if method.lower()[:3]  == 'ave':
                target_array = target_array / (i+1)
                
            monthly_accumulated_array = monthly_accumulated_array + target_array
        
        annual_average_array = monthly_accumulated_array / len(months)
        annual_average_array = annual_average_array - 273.15
        return annual_average_array

    def CovariateGenerator(self, func, start_date, end_date, t1_key, t2_key=None, threshold_temp_1=None, threshold_temp_2=None): 
        
        '''
        A general function of creating climatic covariates for multiple years.
        
        func:             a keyword to determine which covariate is going to be created
        start_date:       the start date of a certain period when the target covariate is matter
        end_date:         the end date of a certain period when the target covariate  is matter
        t1_key:           key words of temperature climate data subdir name (or filename)
        t2_key:           key words of another temperature climate data subdir name (or filename)
        threshold_temp_1: the base temperature threshold of selected crop to calculate 
                          the target covariate
        threshold_temp_2: another base temperature threshold of selected crop to calculate 
                          the target covariate           
        '''
        
        t1_dict = self.__GetFileDictionary__(start_date, end_date, t1_key)
        total_years = len(t1_dict)
        
        if t2_key is not None: 
            t2_dict = self.__GetFileDictionary__(start_date, end_date, t2_key)
        
        covariate_array = np.zeros(self.ref_array.shape)
    
        for year in t1_dict:
            annual_array = np.zeros(self.ref_array.shape)
            try:
                if func == 'frost':
                    annual_array = self.__AnnualTemperatrueFrequency__(t1_dict[year], threshold_temp_1, 'below')          
                elif func == 'GDD':
                    annual_array = self.__AnnualGDD__(t1_dict[year], t2_dict[year], threshold_temp_1)
                elif func == 'chill':
                    annual_array = self.__AnnualChillHours__(t1_dict[year], t2_dict[year], threshold_temp_1, threshold_temp_2)
                elif func == 'mean_max_monthly_temp':
                    annual_array = self.__AnnualMeanMonthlyBasedOnDaily__(year, t1_dict[year], 'max')
                elif func == 'avg_daily_max_temp_flowering':
                    annual_array = self.__AnnualMeanDaily__(t1_dict[year])
                elif func == 'daily_max_temp_ripening':
                    annual_array = self.__AnnualTemperatrueFrequency__(t1_dict[year], threshold_temp_1, 'above')
            except:
                pass

            covariate_array = covariate_array + annual_array 
            
        covariate_array = covariate_array / total_years
        covariate_array = np.where(self.ref_array == self.no_data, self.no_data, covariate_array)
        
        return covariate_array

class ClimaticCovariate(ClimaticCovariates):
    
    def GetFrostRiskArray(self, tmin_key, start_date, end_date, threshold_tmp):
        func = 'frost'
        return self.CovariateGenerator(func, start_date, end_date, tmin_key, threshold_temp_1=threshold_tmp)

    def GetGDDArray(self, tmin_key, tmax_key, start_date, end_date, threshold_tmp): 
        func = 'GDD'
        return self.CovariateGenerator(func, start_date, end_date, tmin_key, tmax_key, threshold_tmp)

    def GetChillHoursArray(self, tmin_key, tmax_key, start_date, end_date, threshold_tmp_min, threshold_tmp_max):
        func = 'chill'
        return self.CovariateGenerator(func, start_date, end_date, tmin_key, tmax_key, threshold_tmp_min, threshold_tmp_max)
    
    def GetMeanMonthlyMaxTemp(self, tmax_key, start_date, end_date):
        func = 'avg_daily_max_temp_flowering'
        return self.CovariateGenerator(func, start_date, end_date, tmax_key)

    def GetMeanDailyMaxTempFlowering(self, tmax_key, start_date, end_date):
        func = 'avg_daily_max_temp_flowering'
        return self.CovariateGenerator(func, start_date, end_date, tmax_key)
    
    def GetDailyMaxTempFrequencyRipening(self, tmax_key, start_date, end_date, threshold_tmp):
        func = 'daily_max_temp_ripening'
        return self.CovariateGenerator(func, start_date, end_date, tmax_key, threshold_temp_1=threshold_tmp)
    
    

    

def getYearList(start_year, end_year):
    '''
    Return a year list based on the given start and end year
    '''
    year_list = []
    y = start_year
    while y <= end_year:
        year_list.append(str(y))
        y += 1
    return year_list

class ConfigParameters(object):
    '''
    This class is developed for getting the parameter values from config.ini file
    '''
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config._interpolation = configparser.ExtendedInterpolation()
        self.config.read_file(codecs.open(config_file, 'r', 'utf8'))
    
    def GetProjectParams(self, proj_header):
        num_crops = int(self.config.get(proj_header,'num_crops'))
        climate_data_dir = self.config.get(proj_header,'climateSource')
        out_dir = self.config.get(proj_header,'outDir')
        start_year = int(self.config.get(proj_header,'start_year'))
        end_year = int(self.config.get(proj_header,'end_year'))
        key_min = self.config.get(proj_header,'key_min_temperature')
        key_max = self.config.get(proj_header,'key_max_temperature')
        covariate_dict = {c.split('_')[-1]:self.config.get(proj_header,c) for c in self.config[proj_header].keys() if c.split('_')[0] == 'covariate'}
        return num_crops, climate_data_dir, out_dir, start_year, end_year, key_min, key_max, covariate_dict
        
    def GetCropName(self, section):
        return self.config.get(section,'crop')
    
    def GetParamsList(self, section, keyword):
        return [self.config.get(section,p) for p in self.config[section].keys() if p.split('_')[0] == keyword]
    
    def GetCropCovariateIndexList(self, section):
        return sorted(list(set([ci.split('_')[0] for ci in self.config[section].keys() if ci != 'crop'])))


if __name__ == '__main__':
    
    conf = input('Please enter the full dir and filename of config file\n(Enter): ')
    
    while not os.path.isfile(conf):
        print('The config.ini file does not exist. Would you like to use the default file?')
        is_default = input('(Yes/No): ')
        if is_default[0].lower() == 'y': 
            conf = r'D:\LandSuitability_AG\data\config\config.ini'
        else:
            conf = input('Please enter the full dir and filename of config file.\nOr leave it blanket to point to the default file\n (Enter): ')
    
    abstemp = 273.15
    config_params = ConfigParameters(conf)
    proj_header = 'projectConfig'
    num_crops, climate_data_dir, out_dir, s_y, e_y, key_min, key_max, covariate_dict = config_params.GetProjectParams(proj_header)
    
    covariate_dir = join(out_dir, 'climatic_covariates')
    if not os.path.exists(covariate_dir):
        os.makedirs(covariate_dir, exist_ok = True)
    
    year_list = getYearList(s_y, e_y)    
    climate_covariate = ClimaticCovariate(year_list, climate_data_dir)
    ref_raster = climate_covariate.ref_raster
    
    out_raster = Raster()
    for i in range(1, num_crops+1):
        crop_header = 'crop_{}'.format(i)
        crop = config_params.GetCropName(crop_header)
        covariate_index_list = config_params.GetCropCovariateIndexList(crop_header)
        print(crop)
        if covariate_index_list:
            for c_index in covariate_index_list:
                print('Generate {} at {} ...'.format(covariate_dict[c_index], dt.datetime.now()))
                param_list = config_params.GetParamsList(crop_header, c_index)
                if c_index == '1' or c_index == '2': # Frost frequency
                    num_required_params = 3
                    if len(param_list) == num_required_params:
                        covariate_array = climate_covariate.GetFrostRiskArray(key_min, param_list[0], param_list[1], float(param_list[2]) + abstemp)
                    else:
                        print('Error: {} did not generated! {} parameters are required, but only {} were provided!'.format(covariate_dict[c_index], num_required_params, len(param_list)))
            
                elif c_index == '3': #GDD
                    num_required_params = 3
                    if len(param_list) == num_required_params:
                        covariate_array = climate_covariate.GetGDDArray(key_min, key_max, param_list[0], param_list[1], float(param_list[2]) + abstemp)
                    else:
                        print('Error: {} did not generated! {} parameters are required, but only {} were provided!'.format(covariate_dict[c_index], num_required_params, len(param_list)))
                
                elif c_index == '4': # Chill hours
                    num_required_params = 4
                    if len(param_list) == num_required_params:
                        covariate_array = climate_covariate.GetChillHoursArray(key_min, key_max, param_list[0], param_list[1], int(param_list[2]) + abstemp, float(param_list[3]) + abstemp)
                    else:
                        print('Error: {} did not generated! {} parameters are required, but only {} were provided!'.format(covariate_dict[c_index], num_required_params, len(param_list)))
                    
                elif c_index == '5': # mean max monthly temperature
                    num_required_params = 2
                    if len(param_list) == num_required_params:
                        covariate_array = climate_covariate.GetMeanMonthlyMaxTemp(key_max, param_list[0], param_list[1])
                    else:
                        print('Error: {} did not generated! {} parameters are required, but only {} were provided!'.format(covariate_dict[c_index], num_required_params, len(param_list)))
                
                elif c_index == '6': # mean max daily temperature flowering
                    num_required_params = 2
                    if len(param_list) == num_required_params:
                        covariate_array = climate_covariate.GetMeanDailyMaxTempFlowering(key_max, param_list[0], param_list[1])
                    else:
                        print('Error: {} did not generated! {} parameters are required, but only {} were provided!'.format(covariate_dict[c_index], num_required_params, len(param_list)))
                
                elif c_index == '7': # max daily temperature frequency ripening
                    num_required_params = 3
                    if len(param_list) == num_required_params:
                        covariate_array = climate_covariate.GetDailyMaxTempFrequencyRipening(key_max, param_list[0], param_list[1], float(param_list[2]) + abstemp)
                    else:
                        print('Error: {} did not generated! {} parameters are required, but only {} were provided!'.format(covariate_dict[c_index], num_required_params, len(param_list)))
                
                    
                out_raster_file = join(covariate_dir, '{}_{}_{}_{}.tif'.format(crop, covariate_dict[c_index], s_y, e_y))
                out_raster.array2Raster(covariate_array, ref_raster, out_raster_file)
                
    print('Finished at {} ...'.format(dt.datetime.now()))
        
        
        