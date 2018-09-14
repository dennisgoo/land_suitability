# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:31:27 2018

This script is for SLMACC project. It creates climatic covariates (annual & monthly mean temperature) for land suitability modelling based on ERA daily climate data.

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


class readRaster(object):
    '''
    Read raster file and return raster related information, e.g. raster array, nodata value etc..
    '''
    def __init__(self):
        pass
#        try:
#            r=gdal.Open(file)
#        except:
#            SystemExit("No such {} file exists!!!".format(file))
#    
#        self.band=r.GetRasterBand(1)
#        r=None
        
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


class extractTimeInfo(object):
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
        
        '''
        # In NZ winter season is around July. Here redefine a consecutive year start from July and end in next June
        fstHalfYear = ['07','08','09','10','11','12']
        scdHalfYear = ['01','02','03','04','05','06']
        dates1 = sorted(list(d.split('.')[0] for d in self.file if d.split('-')[0] == year and d.split('.')[-1] == 'tiff' and d.split('-')[1] in fstHalfYear))
        dates2 = sorted(list(d.split('.')[0] for d in self.file if d.split('-')[0] == str(int(year)+1) and d.split('.')[-1] == 'tiff' and d.split('-')[1] in scdHalfYear))
        if len(dates2) > 0:
            dates = dates1 + dates2
            return dates
        else:
            return []
        '''
        
class rasterAggregation(object):
    
    def __init__(self, path, mask_array, noDataValue):
        
        self.path = path
        self.mask = mask_array
        self.noData = noDataValue
        self.dates = []
        
    def stackAverage(self, dates):
        self.dates = dates
        i = 0
        for date in self.dates:
            rst = join(self.path, "{}.tif".format(date))
            readrst = readRaster()
            dataArray = readrst.getRasterArray(rst)
            
            if i == 0:                
                aggregatedArray = dataArray
                i+=1
            else:
                aggregatedArray = np.add(aggregatedArray, dataArray)
                i+=1
        
        avgArray = np.where(self.mask == 1, aggregatedArray/i, self.noData)
        
        return avgArray
        
def createMaskArray(raster_array, noData):
    
    mask_array = np.where(raster_array == noData, 0, 1)
    return mask_array

def Array2Raster(inArray, refRaster, newRaster, NoData_value):
    
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

def MeanFromMinMax(path, mask_array, nodata, referenceRaster):
    
    maxtemp = 'MaxTemp'
    mintemp = 'MinTemp'
    
    maxfiles = []
    minfiles = []
    
    for (dirpath, dirnames, filenames) in walk(path):
        for f in filenames:
            if maxtemp in f:
                maxfiles.append(join(path, f))
            elif mintemp in f:
                minfiles.append(join(path, f))
    
    ra = rasterAggregation(path, mask_array, nodata)
    for maxfile in maxfiles:
        for minfile in minfiles:
            if maxfile.split('_')[-1] == minfile.split('_')[-1]:
                avg_temp_array = ra.stackAverage([maxfile.split('.')[0], minfile.split('.')[0]])
                outRaster = 'annual_avg_Temp_{}'.format(maxfile.split('_')[-1])                                            
                outFile = join(path, outRaster)
                Array2Raster(avg_temp_array, referenceRaster, outFile, noDataValue)
    
       

if __name__ == '__main__':
    climDataPath = r'D:\LandSuitability_AG\data\climate\nz'
    dailyPath = join(climDataPath, 'daily')
    annualPath = join(climDataPath, 'annual')
    if not os.path.exists(annualPath):
        os.makedirs(annualPath, exist_ok = True)
    monthlyPath = join(climDataPath, 'monthly')
    if not os.path.exists(monthlyPath):
        os.makedirs(monthlyPath, exist_ok = True)
    
    referenceRaster = join(dailyPath, 'MaxTempCorr_VCSN_xaavh_1971-1980', '1971-01-01.tif')
    readrst = readRaster()
    noDataValue = readrst.getNoDataValue(referenceRaster)
    ref_array = readrst.getRasterArray(referenceRaster)
    
    mask_array = createMaskArray(ref_array, noDataValue)
    
    #------Calculate mean temperature based on min and max temperature-----------
    MeanFromMinMax(annualPath, mask_array, noDataValue, referenceRaster)
    #----------------------------------------------------------------------------
    
    for root, subroots, files in walk(dailyPath):
        for subroot in subroots:
            print('Start to process {} at {}'.format(subroot, dt.datetime.now()))
            for rt, sbrs, fs in walk(join(root, subroot)):
                timeExtractor = extractTimeInfo(join(root, subroot))
                years = timeExtractor.extractYears()
                for y in years:
                    print('Start to process year {} at {}'.format(y, dt.datetime.now()))
                    
                    # daily climate files in each year
                    dates = timeExtractor.extractDates(y)
                    raster_aggregator = rasterAggregation(join(root, subroot), mask_array, noDataValue)
                    annual_avg_array = raster_aggregator.stackAverage(dates)
                    
                    outRaster = 'annual_avg_{}_{}.tif'.format(subroot.split('_')[0], y)
                    outFile = join(annualPath, outRaster)
                    Array2Raster(annual_avg_array, referenceRaster, outFile, noDataValue)
                    
                    # Create monthly aggregation
                    months = timeExtractor.extractMonths(y)
                    for m in months:
                        dates = timeExtractor.extractDates(y, m)
                        monthly_avg_array = raster_aggregator.stackAverage(dates)
                        
                        outRaster = 'monthly_avg_{}_{}_{}.tif'.format(subroot.split('_')[0], y, m)                                            
                        outFile = join(monthlyPath, outRaster)
                        Array2Raster(monthly_avg_array, referenceRaster, outFile, noDataValue)
                    
    
    
    
                