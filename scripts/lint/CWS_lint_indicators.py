import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import numpy as np
from datetime import datetime

# Calculate LINT-based drought indicators
# (calculate LINT and SMA zscores, SMA-gs masked, SMA indicators first)


""" Calculate drought impact indicator

Parameters:
    aoi_name: str
        Area of interest
    aoi_coords: tuple (xmin, ymin, xmax, ymax)
        Bounding box coordinates of AOI
    year: int
        Year for which to calculate indicators
    base_path: path object
        Path to folder that contains SMA and LINT folders
"""
base_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought')

# bounding box coordinates given in (xmin, ymin, xmax, ymax)
# aoi_bbox = dict(west=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2,5500000.0000 ),
#                   east=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000,5500000.0000 ))

aoi_bbox = dict(Nwest=(1500000.0000, 900000.0000+(5500000.0000-900000.0000)/2, 1500000.0000+(7400000.0000-1500000.0000)/2, 5500000.0000 ),
                Neast=(1500000.0000+(7400000.0000-1500000.0000)/2, 900000.0000+(5500000.0000-900000.0000)/2, 7400000.0000, 5500000.0000 ),
                Swest=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2, 900000.0000+(5500000.0000-900000.0000)/2 ),                
                Seast=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000, 900000.0000+(5500000.0000-900000.0000)/2))

for aoi_name in list(aoi_bbox):
    aoi_coords = aoi_bbox[aoi_name]
    for year in range(2000, 2022):
        
        ################################################ MR-VPP INDICATORS #################################################
        spacing = 500
        # spacing = 0
        xmin = aoi_coords[0] - spacing
        ymin = aoi_coords[1] - spacing
        xmax = aoi_coords[2] + spacing
        ymax = aoi_coords[3] + spacing

        lint_path = os.path.join(base_path, f'LINT_anom_{aoi_name}')
        out_path = os.path.join(base_path, f'MRVPP_workflow_results_{aoi_name}')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        # load LINT into xarray
        lint_ds = xr.open_dataset(os.path.join(lint_path, f'LINT_anom_{year}_s1.tif'), engine='rasterio').sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
        # Rename the variable to a more useful name
        lint_ds = lint_ds.rename({'band_data': 'linta'})
        # Rename dimension/coordinate 'band' to 'time'
        lint_ds = lint_ds.rename(dict(band='time'))
        lint_ds['time'] = [datetime(year, 1, 1)]

        # ANNUAL MEDIUM RES DROUGHT IMPACT
        ann_drought_pressure = xr.open_dataset(os.path.join(base_path, f'MRVPP_workflow_results_{aoi_name}', f'sm_ann_dp_{year}.tif'), engine='rasterio')
        ann_drought_pressure = ann_drought_pressure.rename({'band_data': 'sma'})
        # Rename dimension/coordinate 'band' to 'time'
        ann_drought_pressure = ann_drought_pressure.rename(dict(band='time'))
        ann_drought_pressure['time'] = [datetime(year, 1, 1)]

        lint_dp = xr.where(ann_drought_pressure['sma'] < -1, lint_ds, np.nan)
        del lint_ds
        ann_mr_drought_impact = xr.where(lint_dp < -0.5, lint_dp, np.nan)
        del lint_dp

        ################################################ RESULTS TO TIFF ######################################################
        results = dict(lint_mr_di=ann_mr_drought_impact)
        for res in list(results):
            v = 'linta'
            results[res] = results[res].fillna(-999)
            results[res][v].rio.write_nodata(-999, inplace=True)
            results[res].rio.write_crs("epsg:3035", inplace=True)
            results[res][v].rio.to_raster(os.path.join(out_path, f'{res}_{year}.tif'))

