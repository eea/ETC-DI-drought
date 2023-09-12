import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd

# Calculate long-term average indicators
# 0) Calculate SM and LINT anomalies and annual SMA and LINT indicators first
# 1) Load annual SM- and LINT-based indicators
# 2) Calculate long-term averages/minima


""" Calculate long-term average drought pressure, drought pressure intensity, drought impact

Parameters:
    aoi: str
        Area of interest (name as used in the previously created tif files)
    base_path: path object
        Path to results folder (MRVPP_workflow_results_<aoi>)
"""

base_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought')

for aoi in ['Nwest', 'Neast', 'Swest', 'Seast']:
    indicator_path = os.path.join(base_path, f'MRVPP_workflow_results_{aoi}')
    out_path = indicator_path

    # LONG-TERM AVG DROUGHT PRESSURE
    tif_list = [f for f in os.listdir(indicator_path) if 'sm_ann_dp_' in f]
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[-8:-4]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs contained in the given folder
    xds = xr.concat([xr.open_dataset(os.path.join(indicator_path, i), engine='rasterio') for i in tif_list], dim=time_var)
    # Rename the variable to a more useful name
    xds = xds.rename({'band_data': 'sma'})
    sma_lta_dp = xds.mean(dim='time')
    sma_lta_dp = sma_lta_dp.drop_vars(['band']).squeeze(['band'])

    # LONG-TERM AVG DROUGHT PRESSURE INTENSITY
    tif_list = [f for f in os.listdir(indicator_path) if 'sm_ann_dpi_' in f]
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[-8:-4]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs contained in the given folder
    xds = xr.concat([xr.open_dataset(os.path.join(indicator_path, i), engine='rasterio') for i in tif_list], dim=time_var)
    # Rename the variable to a more useful name
    xds = xds.rename({'band_data': 'sma'})
    sma_lta_dpi = xds.mean(dim='time')
    sma_lta_dpi = sma_lta_dpi.drop_vars(['band']).squeeze(['band'])

    # LONG-TERM AVG DROUGHT IMPACT
    tif_list = [f for f in os.listdir(indicator_path) if 'lint_mr_di_' in f]
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[-8:-4]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs contained in the given folder
    xds = xr.concat([xr.open_dataset(os.path.join(indicator_path, i), engine='rasterio') for i in tif_list], dim=time_var)
    # Rename the variable to a more useful name
    xds = xds.rename({'band_data': 'linta'})
    lint_lta_di = xds.mean(dim='time')
    lint_lta_di = lint_lta_di.drop_vars(['band']).squeeze(['band'])

    ################################################ RESULTS TO TIFF ######################################################
    results = dict(sma_lta_dp=sma_lta_dp, sma_lta_dpi=sma_lta_dpi, lint_lta_di=lint_lta_di)
    for res in list(results):
        if 'sm' in res:
            v = 'sma'
        elif 'lint' in res:
            v = 'linta'
        results[res].rio.write_crs("epsg:3035", inplace=True)
        results[res][v].rio.to_raster(os.path.join(out_path, f'{res}.tif'))


