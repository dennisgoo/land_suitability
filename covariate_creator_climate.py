# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 12:21:31 2018

This script is for SLMACC project. It creates climatic covariates including:
    1. Frost risk
    2. Growing degree days
    3. Chilling hours

@author: guoj
"""


def GetFrostRisk(crop_name, tmin_key, threshold_tmp, cls_dict, start_date, end_date, year_list):
    '''
    Frost risk frequency was determined by counting years 
    that had at least 1 day of frost occuring at less than
    the threshold temperature between a certain period in
    a year. This count was summed and divided by the total 
    number years.
    
    crop_name:     specific crop name
    tmin_key:      key words of minimum temperature climate data filename
    threshold_tmp: the temperature threshold of selected crop to determine
                   the occurance of frost risk (compared with tmin)
    cls_dict:      a dictionary of suitability rule of frost risk
    start_date:    the start date of a certain period when frost risk is matter
    end_date:      the end date of a certain period when frost risk is matter
    year_list:     a numeric list of years which are took into account when calculating
                   frost risk frequency
    '''
    pass