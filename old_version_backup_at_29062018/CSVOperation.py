# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:29:10 2017

Read TAW & RAW data for siblings from CSV file

Write out CSV file base on specific headers

@author: Jing Guo
"""
import csv
from pathlib import Path

class CSVReading:
    
    def __init__(self):
        self.__filename = ''
        self.__data = []
        
    def __OpenFile(self, filename):
        
        file = Path(filename)
        
        try:
            file.resolve()
        except FileNotFoundError:
            print("File " + filename + "does not exist")
        else:
            self.__filename = filename
            
            with open(self.__filename) as f:
                reader = csv.DictReader(f)
                self.__data = list(reader)
            
#            return self.__data
    
    def GetWaterCapacity(self, filename, header):
        self.__header = header
        
        self.__OpenFile(filename)
        
        if len(self.__data) != 0:
            
            # number of realization, the value of 'LayerNo' represent the number of realization
            self.__numRealizations = max(list(map (lambda x:int(x[self.__header]),self.__data)))
                        
            return self.__data, self.__numRealizations 
    

class CSVWriting:
    
    def __init__(self):
        self.__filename = ''

    # Write CSV file line by line, ease line is a list including the header line
    def WriteLines(self, filename, headers, data):
        self.__filename = filename
        self.__headers = headers
        self.__data = data
        
        with open(self.__filename,'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames = self.__headers, delimiter = ',')
            writer.writeheader()
            
            filewriter = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerows(self.__data)
            
            