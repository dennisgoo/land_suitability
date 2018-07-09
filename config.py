# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:34:27 2018
Read configration file for both covariate generator and suitability mapping

@author: guoj
"""
import configparser
import codecs

class ConfigParameters(object):
    '''
    This class is developed for getting the parameter values from config.ini file
    '''
    def __init__(self, config_file):
        
        self.config = configparser.ConfigParser()
        self.config._interpolation = configparser.ExtendedInterpolation()
        self.config.read_file(codecs.open(config_file, 'r', 'utf8'))
    
    def GetDB(self, proj_header):
        
        db = self.config.get(proj_header,'db')
                
        return db
        
    def GetClimateScenario(self, proj_header):
        
        cli_scenario_dict = {}
        cli_scenario_dict['era'] = self.config.get(proj_header,'era')
        cli_scenario_dict['rcp26'] = self.config.get(proj_header,'rcp26')
        cli_scenario_dict['rcp45'] = self.config.get(proj_header,'rcp45')
        cli_scenario_dict['rcp60'] = self.config.get(proj_header,'rcp60')
        cli_scenario_dict['rcp85'] = self.config.get(proj_header,'rcp85')
        
        return cli_scenario_dict
    
    def GetClimateModel(self, proj_header):
        
        cli_model_dict = {}
        cli_model_dict['bcc'] = self.config.get(proj_header,'model_bcc')
        cli_model_dict['cesm'] = self.config.get(proj_header,'model_cesm')
        cli_model_dict['gfdl'] = self.config.get(proj_header,'model_gfdl')
        cli_model_dict['giss'] = self.config.get(proj_header,'model_giss')
        cli_model_dict['hadgem'] = self.config.get(proj_header,'model_hadgem')
        cli_model_dict['noresm'] = self.config.get(proj_header,'model_noresm')
        
        return cli_model_dict
    
    def GetClimateCovariateDir(self, proj_header):
        
        # This is the dir where the climate covariates generated from 'covariate_creator_climate_v3.py' locate.
        cova_dir = self.config.get(proj_header,'climate_covariates_dir')
        
        return cova_dir
    
    def GetSoilCovariateDir(self, proj_header):
        
        cova_dir = self.config.get(proj_header,'soil_covariates_dir')
        
        return cova_dir
    
    def GetTerrainCovariateDir(self, proj_header):
        
        cova_dir = self.config.get(proj_header,'terrain_covariates_dir')
        
        return cova_dir
    
    def GetProcessedCovariateDir(self, proj_header):
        
        cova_dir = self.config.get(proj_header,'processed_covariates_dir')
        
        return cova_dir
        
    def GetClimateCovariateParams(self, proj_header, cli_header):
        
        climate_dir = self.config.get(proj_header,'climate_dir')
        start_year = int(self.config.get(cli_header,'start_year')) 
        end_year = int(self.config.get(cli_header,'end_year'))
        key_min = self.config.get(cli_header,'key_min_temperature')
        key_max = self.config.get(cli_header,'key_max_temperature')
        
        return climate_dir, start_year, end_year, key_min, key_max
    
        
    def GetPreprocessingParams(self, proj_header, prep_header):
        
        soil_file = self.config.get(proj_header,'soil_file')
        slope_file = self.config.get(proj_header,'slope_file')
        spat_res = self.config.get(prep_header,'resolution')
        
        return soil_file, slope_file, spat_res
    
    def GetSuitabilityParams(self, proj_header):
        
        # This is the dir where the final suitability maps locate.
        suit_map_dir = self.config.get(proj_header,'suitability_map_dir')
        
        return suit_map_dir    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        