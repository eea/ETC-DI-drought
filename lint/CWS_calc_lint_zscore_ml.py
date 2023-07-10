import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
from datetime import datetime
#from pyproj import CRS
import pyproj
import geopandas as gpd
import rasterio


print ("PART00 REading libs")


# Calculate MR-VPP LINT anomalies
# 1) Load all yearly LINT files
# 2) Calculate long-term average and standard deviation over defined base
# 3) Calculate zscores as (annual_LINT - lta_LINT) / stdev_LINT
# 4) Write yearly LINT anomaly files


""" Calculate LINT anomalies: LINTanom = (LINT - LTA) / stdev1
LINTanom...LINT anomaly, LTA...long-term average, stdev...standard deviation

aoi_name...area of interest (must be defined in aoi_coords). Necessary in order to avoid memory error
aoi_coords...bounding box coordinates of AOI
base_path: path object (path to folder that contains LINT folder)
base1, base2...start and end of baseline period

"""

start_year = 2000
end_year = 2022 # last year for which we have data
base_path = Path(r'L:\drought_indicator')
base1 = 2001
base2 = 2020


print ("PART02--bbox")


aoi_bbox = dict(Nwest=(1500000.0000, 900000.0000+(5500000.0000-900000.0000)/2, 1500000.0000+(7400000.0000-1500000.0000)/2, 5500000.0000 ),
                Neast=(1500000.0000+(7400000.0000-1500000.0000)/2, 900000.0000+(5500000.0000-900000.0000)/2, 7400000.0000, 5500000.0000 ),
                Swest=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2, 900000.0000+(5500000.0000-900000.0000)/2 ),                
                Seast=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000, 900000.0000+(5500000.0000-900000.0000)/2))


# for aoi_name in ['Nwest', 'Neast']:
for aoi_name in list(aoi_bbox):
    aoi_coords = aoi_bbox[aoi_name]
    print ("box:")
    print (aoi_coords)
    # AOI: subset of Europe, e.g. Iberian peninsula (in order to get everything into memory)
    # (adding 1x spacing to make sure SMA covers entire AOI)
    spacing = 500
    xmin = aoi_coords[0] - spacing
    ymin = aoi_coords[1] - spacing
    xmax = aoi_coords[2] + spacing
    ymax = aoi_coords[3] + spacing

    baseline_period = (base1, base2)

    lint_path = os.path.join(base_path, 'LINT')
    out_path = os.path.join(base_path, f'LINT_anom_{aoi_name}')
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # load LINT into xarray
    tif_list = [f for f in os.listdir(lint_path) if 'LINT' in f]  
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[12:16]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs
    lint_ds = xr.concat([xr.open_dataset(os.path.join(lint_path, i), engine='rasterio').sel(y=slice(ymax, ymin), x=slice(xmin, xmax)) for i in tif_list], dim=time_var)
    # Rename the variable to a more useful name
    lint_ds = lint_ds.rename({'band_data': 'LINT'})

    # calculate LINT anomaly (difference from long-term average, standardized by stdev -> zscores)
    # baseline period as defined above (default: 2000-2015)
    lint_lta = lint_ds.sel(time=slice(datetime(baseline_period[0], 1, 1), datetime(baseline_period[1], 12, 31))).mean(dim='time')
    lint_std = lint_ds.sel(time=slice(datetime(baseline_period[0], 1, 1), datetime(baseline_period[1], 12, 31))).std(dim='time')
    lint_anom = (lint_ds - lint_lta) / lint_std

    lint_anom = lint_anom.fillna(-999)
    lint_anom = xr.where(lint_std == 0, -999, lint_anom)    # mask pixels where stdev = 0, i.e. anom is inf
    lint_anom['LINT'].rio.write_nodata(-999, inplace=True)


    lint_anom.rio.write_crs("epsg:3035", inplace=True)   #### please check, .!!!!!!!!!!!

   

    for y in range(start_year, end_year+1):
        print (y)
        lint_anom['LINT'].sel(time=datetime(y, 1, 1)).rio.to_raster(os.path.join(out_path,
                                                                                 f'LINT_anom_{y}_s1.tif'),
                                                                    compress='LZW')

print ("end")