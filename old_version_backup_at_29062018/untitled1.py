# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:54:53 2018

@author: guoj
"""

def transfer(x):
    return {
        'Silty clay': 2,
        'Loamy silt': 3,
        'Sandy clay loam': 5,
        'Clay loam': 6,
        'Silt': 7,
        'Silt loam': 8,
        'Sand': 10,
        'Loamy sand': 11,
        'Sandy loam': 12
    }.get(x, 99)
    
def recalculateslope(x):
    
    if x == '>26':
        return (26+35)/2
    else:
        vmin = int(x.split('-')[0])
        vmax = int(x.split('-')[1])
        return (vmax+vmin)/2
    
    
        
        