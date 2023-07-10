# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:10:24 2022

@author: ivits
"""

#Libraries used in the script
import numpy as np
import datetime as dt
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob
import os
import sys


"""
def paths_to_datetimeindex(paths):
        return [dt.strptime(date.split('/') [-2],'%Y') for date in paths]

data_dir='S:/Common workspace/ETC_DI/AP22_Drought/LINT'

scene_directories = glob(os.path.join(data_dir, '*/'))

time_var = xr.Variable('time', paths_to_datetimeindex(scene_directories))
"""                     
                       
                       
data_list=glob('S:/Common workspace/ETC_DI/AP22_Drought/LINT/*.tif')

years=np.arange(2000,2000+len(data_list)) 
                                                
time_var = xr.Variable('time', years)

LINTda=xr.concat([xr.open_rasterio(i) for i in data_list], dim=time_var)

LINTds=LINTda.to_dataset('band')

LINTds=LINTds.rename({1:'LINT'})

