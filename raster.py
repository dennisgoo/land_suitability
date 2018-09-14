# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:50:19 2018
A raster class used to initialize raster object, read and write raster.
@author: guoj
"""
from osgeo import gdal, osr
import numpy as np

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