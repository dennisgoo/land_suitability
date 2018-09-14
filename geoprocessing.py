# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:53:48 2018

@author: guoj
"""

from osgeo import gdal, ogr, osr
from os.path import join
import numpy as np
from raster import Raster

rst = Raster()

class GeoProcessing(object):
    def __init__(self, no_data):
        self.no_data = no_data
    
    def strip_end(self, text, suffix):
        if not text.endswith(suffix):
            return text
        return text[:len(text)-len(suffix)]
    
    def strip_start(self, text, suffix):
        if not text.startswith(suffix):
            return text
        return text[len(suffix):]        
    
    def reproject_raster2(self, dataset, refdataset, outrst):
    
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
    
    def reproject_raster(self, dataset, refdataset, outrst):
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
        outband.SetNoDataValue(self.no_data)
        # Perform the projection/resampling 
        gdal.ReprojectImage(g, dest, wgs84.ExportToWkt(), nz2000.ExportToWkt(), gdal.GRA_NearestNeighbour)
    
    #    return dest
    
    def resample_image_Nearest(self, dataset, refdataset):
        
        
        r_fine = gdal.Open(refdataset) 
        r_coarse = gdal.Open(dataset) 
    
        band = r_coarse.GetRasterBand(1)
        coarse_array = band.ReadAsArray().astype(np.float)
        
        #---------------------------------------------------------------
        #grow coarse raster for 1 pixel based on the nearest value
        cols = r_coarse.RasterXSize
        rows = r_coarse.RasterYSize
        growArray = np.full((rows, cols), float(self.no_data))
        for x in range(0, rows):
            for y in range(0, cols):
                if int(coarse_array[x, y]) == self.no_data:
                    neighbors = []
                    for i in range(x-1, x+2):
                        for j in range(y-1, y+2):
                            try:
                                if int(coarse_array[i,j]) != self.no_data:
                                    neighbors.append(coarse_array[i,j])
                            except:
                                pass
                    if len(neighbors) > 0:
                        growArray[x, y] = np.mean(neighbors)
         
        coarse_array = np.where(coarse_array == self.no_data, growArray, coarse_array)     
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
        
        ref_array = rst.getRasterArray(refdataset)
        newArray = np.where(ref_array == self.no_data, self.no_data, newArray)
        rst.array2Raster(newArray, refdataset, dataset)
    
    def CompareRasterSize(self,raster, refraster):
        r = gdal.Open(raster) 
        r_ref = gdal.Open(refraster) 
    
        r_cols = r.RasterXSize
        r_rows = r.RasterYSize
        
        r_ref_cols = r_ref.RasterXSize
        r_ref_rows = r_ref.RasterYSize
        
        if (r_cols != r_ref_cols) or (r_rows != r_ref_rows):
            self.resample_image_Nearest(raster, refraster)
        else:
            pass
    
    def RemoveEmptyRowsCols(self, np_array):
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
        
    def ESRIVector2Raster(self, file, field_id, field_Name, raster_res, out_dir):
        
        raster_res = int(raster_res)
        
        try:
            if file.split('.')[-1].lower() == 'shp':
                driver = ogr.GetDriverByName("ESRI Shapefile")
                ds = driver.Open(file, 0)
                layer = ds.GetLayer()  
            else:
                driver = ogr.GetDriverByName("OpenFileGDB")
                layer_name = file.split('\\')[-1]
                gdb = self.strip_end(file, layer_name)
                ds = driver.Open(gdb, 0)
                layer = ds.GetLayer(layer_name)
        except:
            layer = None
        
        if layer is not None:
            feature = layer.GetNextFeature()
            geometry = feature.GetGeometryRef()
            geoName = geometry.GetGeometryName()
            fieldType = self.GetLayerFieldType(layer, field_Name)
            
    
            if fieldType is not None:
                if geoName == 'MULTIPOLYGON' or geoName == 'POLYGON':
                    out_raster_file = join(out_dir, '{}.tif'.format(field_id))
                    self.PolygonToRaster(layer, out_raster_file, field_Name, raster_res, self.DataTypeConversion(fieldType))
                else:
                    print('"{}" is not a polygon layer!'.format(file))
            else:
                print('Field "{}" does not exist in layer "{}"!'.format(field_Name, file))
        else:
            print('Layer "{}" does not exist!'.format(file))
            
    
    def GetLayerFieldType(self, layer, field_name):
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
    def PolygonToRaster(self, inLayer, outFile, attr, pixel_size, d_type):
        
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
        band.SetNoDataValue(self.no_data)
    
        # Rasterize
        gdal.RasterizeLayer(target_ds, [1], inLayer, options = ["ATTRIBUTE="+attr])
        
    #    print('{} {}'.format(attr, d_type))
        inLayer = None
        target_ds = None
        band = None
    
    def DataTypeConversion(self, x):
        return {
                'Real': gdal.GDT_Float32,
                'Integer': gdal.GDT_Int32,
            }.get(x, gdal.GDT_Unknown)
        
    def DataTypeConversion_GDAL2NP(self, x):
        return {
                5: np.int,
                6: np.float,
            }.get(x, np.float)
