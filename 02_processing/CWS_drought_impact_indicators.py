import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np
import geopandas as gpd
from loguru import logger

# 0) Calculate drought impact mask, based on i) annual gs SMA and ii) annual GDMP anomalies (SMA <-1 & GDMPA < 0)
# 1) Calculate drought impact intensity, based on i) drought impact mask and ii) annual GDMP deviations

@logger.catch
def calc_drought_impact_indicators(year, base_path):
    logger.info(f"Computing year {year}")
    SMA_path = os.path.join(base_path, "drought_indicator", "output", "SMA_gs_1km_EU", "SMA_gs_1km_annual")
    GDMP_path = os.path.join(base_path, "GDMP", "03_output", "GDMPa_1km")

    ############################################ drought impact mask ###################################################
    sma_filename = f"SMA_gs_1km_annual_{year}.tif"
    sma_xr = rio.open_rasterio(os.path.join(SMA_path, sma_filename))
    sma_xr = xr.where(sma_xr==-999, np.nan, sma_xr)

    gdmpa_filename = f"GDMP_1km_anom_{year}.tif"
    gdmpa_xr = rio.open_rasterio(os.path.join(GDMP_path, gdmpa_filename))
    gdmpa_xr.sel(x=np.intersect1d(sma_xr.x.values, gdmpa_xr.x.values),
                 y=np.intersect1d(sma_xr.y.values, gdmpa_xr.y.values))

    # create the drought impact mask: assign 1 where SMA<-1 and GDMPa<0, assign np.nan elsewhere
    drought_impact_mask = xr.where(((sma_xr<-1) & (gdmpa_xr<0)), 1, np.nan)
    drought_impact_mask.rio.write_nodata(-999, inplace=True)
    drought_impact_mask.rio.write_crs("epsg:3035", inplace=True)
    drought_impact_mask.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    out_path = os.path.join(base_path, "GDMP", "03_output", "drought_impact_mask_gs_1km")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger.info("Saving drought impact mask")
    drought_impact_mask.rio.to_raster(os.path.join(out_path, "drought_impact_mask_"+str(year)+".tif"))

    ######################################### drought impact intensity #################################################
    gdmpa_filename = f"GDMP_1km_deviation_{year}.tif"
    gdmpa_xr = rio.open_rasterio(os.path.join(GDMP_path, gdmpa_filename))
    gdmpa_xr = gdmpa_xr.sel(x=np.intersect1d(drought_impact_mask.x.values, gdmpa_xr.x.values),
                 y=np.intersect1d(drought_impact_mask.y.values, gdmpa_xr.y.values))

    # create the drought impact intensity layer: assign yearly GDMP deviation where annual SMA<-1
    drought_impact_intensity = gdmpa_xr.where(drought_impact_mask == 1)
    drought_impact_intensity.rio.write_nodata(-999, inplace=True)
    drought_impact_intensity.rio.write_crs("epsg:3035", inplace=True)
    drought_impact_intensity.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    out_path = os.path.join(base_path, "GDMP", "03_output", "drought_impact_intensity_gs_1km")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger.info("Saving drought impact intensity")
    drought_impact_intensity.rio.to_raster(os.path.join(out_path, "drought_impact_intensity_"+str(year)+".tif"))

    return None


if __name__ == "__main__":

    # add log file
    logger.add("99_logfiles\logfile_drought_impact_indicator.log")

    base_path = Path(r'L:\f02_data')
    aoi_coords = dict(EU=(2500000, 750000, 7500000, 5500000))

    for year in range(2000, 2023):
        calc_drought_impact_indicators(year=year, base_path=base_path)
