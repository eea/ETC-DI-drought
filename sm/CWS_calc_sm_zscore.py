import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
from datetime import datetime

# Calculate JRC SMI anomalies
# 1) Load all yearly SMI files
# 2) Calculate long-term average and standard deviation over defined baseline period (for each dekad)
# 3) Calculate zscores as (SMI_dekad - lta_dekad) / stdev_dekad
# 4) Write yearly SM anomaly files (.tif) with fill value -999
# Note: no east/west AOI splitting needed due to coarse spatial resolution


""" Calculate SM anomalies: SMA_dy = (SMI_dy - LTA_d) / stdev_d
SMA...sm anomaly, SMI...sm index, LTA...long-term average, stdev...standard deviation
dy...dekad x of year y
d...dekad x of all years in baseline period

base_path: path object (path to folder that contains JRC_SMI folder)
base1, base2...start and end of baseline period
"""

start_year = 2000
end_year = 2022 # last year for which we have data
base_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought')
base1 = 2001
base2 = 2020

baseline_period = (base1, base2)
smi_path = os.path.join(base_path, 'JRC_SMI')
○out_path = os.path.join(base_path, 'JRC_SMA_custom')
if not os.path.exists(out_path):
    os.mkdir(out_path)

smi = xr.open_mfdataset(os.path.join(smi_path, '*.nc'))
# group by month and day to get long-term average and standard deviation for each dekad
month_day_idx = pd.MultiIndex.from_arrays([smi['time.month'].values, smi['time.day'].values])
smi.coords['month_day'] = ('time', month_day_idx)
dekadal_lta = smi.sel(time=slice(datetime(baseline_period[0], 1, 1), datetime(baseline_period[1], 12, 31))).groupby('month_day').mean()
dekadal_std = smi.sel(time=slice(datetime(baseline_period[0], 1, 1), datetime(baseline_period[1], 12, 31))).groupby('month_day').std()
# avoid data gaps (division by std=0) by setting values equal to 0 to 0.01
dekadal_std = xr.where(dekadal_std == 0, 0.01, dekadal_std)

# calculate standardized anomalies and save to annual SMA files
for year in range(start_year, end_year+1):
    smi_yr = smi.sel(time=slice(datetime(year, 1, 1), datetime(year, 12, 31)))
    smi_yr = smi_yr.assign(dekadal_lta=(('time', 'lat', 'lon'), dekadal_lta['sminx'].values))
    smi_yr = smi_yr.assign(dekadal_std=(('time', 'lat', 'lon'), dekadal_std['sminx'].values))
    sma_zscore = (smi_yr['sminx'] - smi_yr['dekadal_lta']) / smi_yr['dekadal_std']
    sma_yr = xr.Dataset(coords=dict(time=smi_yr.time, lat=smi_yr.lat, lon=smi_yr.lon),
                        data_vars=dict(sma=(('time', 'lat', 'lon'), sma_zscore.values)))

    sma_yr = sma_yr.fillna(-999)
    sma_yr['sma'].rio.write_nodata(-999, inplace=True)
    sma_yr.rio.write_crs("epsg:3035", inplace=True)
    sma_yr['sma'].rio.to_raster(os.path.join(out_path, f'SMI_anom_{year}.tif'), compress='LZW')

