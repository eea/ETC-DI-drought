import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator, RegularGridInterpolator
from datetime import datetime
import matplotlib.pyplot as plt
import calendar
import itertools
from datetime import datetime
from loguru import logger


# Calculate growing-season masked SMA
# 0) Calculate SMA zscores first -> CWS_calc_sm_zscore.py
# 1) Reformat SOS and EOS (YYDOY -> DOY) using reclassify_dsos/eos.ipynb
# 2) Load SOS and EOS dates, reformat
# 3) Load SM anomalies, resample to 1km grid, mask outside of growing season
# 4) Save dekadal, growing-season masked SMA files (from earliest SOS date to latest EOS date in study region)
# 5) Derive and save drought indicators

@logger.catch
def resample_sm(sma_in, src_y, src_x, target_y, target_x, k=1):
    """ Resample SMA from source grid (5 km) to target grid (1 km) using bivariate spline
    """
    # replace NaNs with default SMA value (0)
    data_arr = np.nan_to_num(np.flipud(sma_in), nan=0)  # x and y coords must be monotonically increasing -> flip upside down

    # calculate bivariate spline function on coarse grid
    interp_spline = RectBivariateSpline(src_y[::-1], src_x, data_arr, kx=k, ky=k)

    # apply spatial interpolation on fine grid
    sma_out = np.flipud(interp_spline(target_y[::-1], target_x))

    return sma_out

@logger.catch
def calc_sma_gs_indicators(aoi_name, aoi_coords, year, base_path):
    """ Input:
    - 5 km SM anomalies (calculate in CWS_calc_sm_zscore.py)
    - 1 km SOS and EOS dates

    1) resample SMA to SOS/EOS grid
    2) mask SMA outside growing season (keep values between SOS and EOS dates)
    3) write dekadal SMA-GS to tif-file

    Parameters:
        aoi_name: str
            Area of interest
        aoi_coords: tuple (xmin, ymin, xmax, ymax)
            Bounding box coordinates of AOI
        year: int
            Year for which to calculate SMA-GS. SOS can be before Jan 1, thus, SMA from previous and current years is
            used. Exception: year 2000 (no SMA and SOS/EOS data before that year)
        base_path: path object
            Path to folder that contains SMA, EOS, SOS folders

    """
    # AOI: subset of Europe, e.g. Iberian peninsula (in order to get everything into memory)
    # (adding 1x spacing to make sure SMA covers entire AOI)
    spacing = 0
    xmin = aoi_coords[0] - spacing
    ymin = aoi_coords[1] - spacing
    xmax = aoi_coords[2] + spacing
    ymax = aoi_coords[3] + spacing

    sos_path = os.path.join(base_path, "input", 'SOS_1km')
    eos_path = os.path.join(base_path, "input", 'EOS_1km')
    sma_path = os.path.join(base_path, "input", 'JRC_SMA_custom')
    # out_path = os.path.join(base_path, f'JRC_SMA_GS_{aoi_name}')
    # ISI code --- out_path = os.path.join(base_path, f'MRVPP_workflow_results_{aoi_name}')
    out_path = os.path.join(base_path, "output", f'SMA_gs_1km_{aoi_name}')
    sm_var = 'sma'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    logger.info("loading eu mask")
    # read shp to mask output files
    shp_path = Path(r"L:\f02_data\ref_data\terrestrial_europe_mask")
    shp_aoi = gpd.read_file(os.path.join(shp_path, "admin_mask_europe.shp"))

    ################################################## SOS & EOS #######################################################
    logger.info("Preparing EOS and SOS")
    # note: SOS ranges from <0 to +365 (previous year + current year)
    # load SOS dates of desired year into xarray
    tif_list = [f for f in os.listdir(sos_path) if str(year) in f]
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[20:24]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs contained in the given folder
    sos_ds = xr.concat([xr.open_dataset(os.path.join(sos_path, i), engine='rasterio') for i in tif_list], dim=time_var).sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
    # Rename to something meaningful
    sos_ds = sos_ds.rename({'band_data': 'dSOS'})
    # create spatial mask where SOS has no data (to be applied on SM later)
    gs_mask = xr.where(sos_ds['dSOS'].isnull(), 1, 0)
    # replace no data by 0 and convert to integer -> needs less memory
    sos_ds = xr.where(sos_ds['dSOS'].isnull(), 0, sos_ds).astype(int)

    # note: EOS ranges from 0 to >365 (current year + following year)
    # load EOS dates of desired year into xarray
    tif_list = [f for f in os.listdir(eos_path) if str(year) in f]
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[20:24]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs contained in the given folder
    eos_ds = xr.concat([xr.open_dataset(os.path.join(eos_path, i), engine='rasterio') for i in tif_list], dim=time_var).sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
    # Rename to something meaningful
    eos_ds = eos_ds.rename({'band_data': 'dEOS'})
    # add spatial EOS mask
    gs_mask = xr.where((gs_mask == 1) | (eos_ds['dEOS'].isnull()), 1, 0)
    # replace 0s by 365
    eos_ds = xr.where(eos_ds == 0, 365 + calendar.isleap(year), eos_ds)
    # if EOS falls into next year, set to Dec. 31st
    eos_ds = xr.where(eos_ds > 365 + calendar.isleap(year), 365 + calendar.isleap(year), eos_ds)
    # replace no data values with 0 and convert to integer
    eos_ds = xr.where(eos_ds['dEOS'].isnull(), 0, eos_ds).astype(int)

    ######################################### SM in growing season (2001 onwards) ######################################
    logger.info("Start SMA processing: clip to growing season and clip to -3,3")
    logger.info(f"Processing year {year}")
    if year != 2000:
        # load soil moisture anomalies into xarray (previous and current year)
        sma5k_year = xr.open_dataset(os.path.join(sma_path, f'SMI_anom_{year}.tif'), engine='rasterio').sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
        sma5k_prevyear = xr.open_dataset(os.path.join(sma_path, f'SMI_anom_{year-1}.tif'), engine='rasterio').sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
        sma5k = xr.concat([sma5k_prevyear, sma5k_year], dim='band')
        # Rename the variable to a more useful name
        sma5k = sma5k.rename({'band_data': sm_var})
        # clamp to (-3, 3)
        sma5k = xr.where(sma5k[sm_var] < -3, -3, sma5k)
        sma5k = xr.where(sma5k[sm_var] > 3, 3, sma5k)
        # define coordinates for resampling
        src_x = sma5k.x
        src_y = sma5k.y
        target_x = sos_ds.x
        target_y = sos_ds.y
        ds_time = [datetime(y, m, d) for y, m, d in list(itertools.product([year-1, year], range(1, 13), [1, 11, 21]))]


        logger.info("Resampling 5km to 1km")
        # resample SMA from 5km to 1km using nearest neighbor (in xarray) ----------------------------------------------
        sma1km = sma5k.interp(x=target_x, y=target_y, method="nearest")
        sma_ds = xr.Dataset(
                            data_vars=dict(sma=(['time', 'y', 'x'], sma1km["sma"].values)),
                            coords=dict(
                                time=ds_time,
                                y=target_y,
                                x=target_x,
                            ),
                            attrs=dict(description="JRC SMA resampled to 1km grid using nearest neighbor interpolation."),
                        )

        """# resample and interpolate each timestamp
        sma1km = np.empty((len(ds_time), len(target_y), len(target_x)))
        for tsi in range(len(ds_time)):
            sma_tsi = resample_sm(sma5k.isel(band=tsi)[sm_var].values, src_y, src_x, target_y, target_x)
            sma1km[tsi] = sma_tsi
        # store in new xarray dataset
        del sma5k, sma_tsi
        sma_ds = xr.Dataset(
                            data_vars=dict(sma=(['time', 'y', 'x'], sma1km)),
                            coords=dict(
                                time=ds_time,
                                y=target_y,
                                x=target_x,
                            ),
                            attrs=dict(description="JRC SMA resampled to 1km grid using nearest neighbor interpolation."),
                        )
        """

        # apply GS masking on SM
        # look-up table for DOY and dekads
        doys_lut = []
        for yi, y in enumerate([year-1, year]):
            for m in range(0, 12):
                _, ndays = calendar.monthrange(y, m+1)
                # for each day in a month, add the corresponding dekad (w.r.t. year-1)
                m_dek = [m*3+1+yi*36]*10 + [m*3+2+yi*36]*10 + [m*3+3+yi*36]*(ndays-20)
                doys_lut += m_dek
        doys_lut = np.array(doys_lut)
        # shift SOS dates to year-1 (to have only positive values, ranging from 1-365*2 for 2 years of data) and
        # convert SOS dates from DOYs to dekads using doys_lut
        sos_dekads = doys_lut[sos_ds['dSOS'] + 365 + calendar.isleap(year-1) - 1]
        del sos_ds

        # shift also EOS by one year to match format of SOS and SMA
        eos_dekads = doys_lut[eos_ds['dEOS'] + 365 + calendar.isleap(year-1) - 1]
        del eos_ds

        # sos_dekads and eos_dekads can now be used for indexing and masking sma_ds
        # (sos_dekad 1 corresponds to index 0, i.e. first timestamp in sma_ds, etc.)
        dekads_arr = np.repeat(np.repeat(np.arange(1, 73)[:, np.newaxis], len(target_y), axis=1)[:, :, np.newaxis], len(target_x), axis=2)
        sma_gs = xr.where((dekads_arr >= sos_dekads)[0] & (dekads_arr <= eos_dekads)[0], sma_ds, np.nan)
        del sma_ds
        # apply mask of no-data SOS/EOS
        gs_mask = gs_mask.drop_vars(['band', 'time']).squeeze(['band', 'time']).expand_dims(dim={'time': len(sma_gs['time'])}).assign_coords(time=sma_gs.time)
        sma_gs = xr.where(gs_mask == 0, sma_gs, np.nan)

    ################################ SM in growing season (exception for year 2000) ######################################
    else:
        # no SMA data before 2000 -> slightly different approach: set all SOS from 1999 to 1.1.2000
        # load soil moisture anomalies into xarray (current year)
        sma5k = xr.open_dataset(os.path.join(sma_path, f'SMI_anom_{year}.tif'), engine='rasterio').sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
        # Rename the variable to a more useful name
        sma5k = sma5k.rename({'band_data': sm_var})
        # clamp to (-3, 3)
        sma5k = xr.where(sma5k[sm_var] < -3, -3, sma5k)
        sma5k = xr.where(sma5k[sm_var] > 3, 3, sma5k)
        # define coordinates for resampling
        src_x = sma5k.x
        src_y = sma5k.y
        target_x = sos_ds.x
        target_y = sos_ds.y
        ds_time = [datetime(year, m, d) for m, d in list(itertools.product(range(1, 13), [1, 11, 21]))]

        # resample and interpolate each timestamp

        logger.info("Resampling 5km to 1km")
        sma1km = sma5k.interp(x=target_x, y=target_y, method="nearest")

        sma1km = np.empty((len(ds_time), len(target_y), len(target_x)))
        for tsi in range(len(ds_time)):
            sma_tsi = resample_sm(sma5k.isel(band=tsi)['sma'].values, src_y, src_x, target_y, target_x)
            sma1km[tsi] = sma_tsi
        # store in new xarray dataset
        del sma5k, sma_tsi
        sma_ds = xr.Dataset(
            data_vars=dict(sma=(['time', 'y', 'x'], sma1km)),
            coords=dict(
                time=ds_time,
                y=target_y,
                x=target_x,
            ),
            attrs=dict(description="JRC SMA resampled to 1km grid using nearest neighbor interpolation."),
        )

        # apply GS masking on SM
        # look-up table for DOY and dekads
        doys_lut = []
        for m in range(0, 12):
            _, ndays = calendar.monthrange(year, m + 1)
            # for each day in a month, add the corresponding dekad (w.r.t. year-1)
            m_dek = [m * 3 + 1] * 10 + [m * 3 + 2] * 10 + [m * 3 + 3] * (ndays - 20)
            doys_lut += m_dek
        doys_lut = np.array(doys_lut)

        # set negative SOS dates to 1st day of year and
        # convert SOS dates from DOYs to dekads using doys_lut
        sos_dekads = doys_lut[sos_ds.where(sos_ds > 0, 1)['dSOS'] - 1]
        del sos_ds

        # set potential negative EOS dates to 1st day of year
        eos_dekads = doys_lut[eos_ds.where(eos_ds > 0, 1)['dEOS'] - 1]
        del eos_ds

        # sos_dekads and eos_dekads can now be used for indexing and masking sma_ds
        # (sos_dekad 1 corresponds to index 0, i.e. first timestamp in sma_ds, etc.)
        dekads_arr = np.repeat(np.repeat(np.arange(1, 37)[:, np.newaxis], len(target_y), axis=1)[:, :, np.newaxis], len(target_x), axis=2)
        sma_gs = xr.where((dekads_arr >= sos_dekads)[0] & (dekads_arr <= eos_dekads)[0], sma_ds, np.nan)
        del sma_ds
        # apply mask of no-data SOS/EOS
        gs_mask = gs_mask.drop_vars(['band', 'time']).squeeze(['band', 'time']).expand_dims(dim={'time': len(sma_gs['time'])}).assign_coords(time=sma_gs.time)
        sma_gs = xr.where(gs_mask == 0, sma_gs, np.nan)

    # free up space
    del sos_dekads, eos_dekads

    # remove timestamps where data is all NaN
    sma_gs = sma_gs.dropna(dim='time', how='all')
    sma_gs = sma_gs.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    sma_gs.rio.write_crs("epsg:3035", inplace=True)
    sma_gs.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    logger.info("saving dekads data")
    ##################### subset SMA_GS to year of interest and save dekadal SMA in yearly folders #####################
    sma_gs_yearly = sma_gs.loc[dict(time=sma_gs.time.dt.year == year)]
    sma_gs_yearly = sma_gs_yearly.fillna(-999)
    # from dataset (time, y, x) to dataset (y, x) and timestamps stored as separate bands
    # needed to store timestamp info in dekadal tiff files
    bands = [pd.to_datetime(str(dt)).strftime('%Y-%m-%d') for dt in sma_gs_yearly.time.values]
    sma_gs_dt = xr.Dataset()
    for bi, band in enumerate(bands):
        sma_gs_dt[band] = xr.DataArray(sma_gs_yearly.isel(time=bi)['sma'].values, coords={'y': sma_gs_yearly.y, 'x': sma_gs_yearly.x})
        sma_gs_dt[band].rio.write_nodata(-999, inplace=True)

    sma_gs_dt.rio.write_crs("epsg:3035", inplace=True)
    sma_gs_dt.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    yearly_out_path = os.path.join(out_path, "SMA_gs_1km_dekadal")
    if not os.path.exists(yearly_out_path):
        os.mkdir(yearly_out_path)

    sma_gs_dt = sma_gs_dt.rio.clip(shp_aoi.geometry.values, all_touched=True)
    sma_gs_dt.rio.to_raster(os.path.join(yearly_out_path, f'SMA_gs_1km_dekadal_{year}.tif'))

    del sma_gs_dt, sma_gs_yearly

    ######################################## calculate and save mean annual SMA ########################################
    logger.info("saving avg annual data")
    annual_sma_gs = xr.where(sma_gs > -4, sma_gs, np.nan).mean("time")
    gs_mask = gs_mask.isel(time=0).drop("time")
    annual_sma_gs = xr.where(gs_mask == 0, annual_sma_gs["sma"], np.nan)
    annual_sma_gs.rio.write_nodata(-999, inplace=True)
    annual_sma_gs.rio.write_crs("epsg:3035", inplace=True)
    annual_sma_gs.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    yearly_out_path = os.path.join(out_path, "SMA_gs_1km_annual")
    if not os.path.exists(yearly_out_path):
        os.mkdir(yearly_out_path)
    annual_sma_gs = annual_sma_gs.rio.clip(shp_aoi.geometry.values, all_touched=True)
    annual_sma_gs.rio.to_raster(os.path.join(yearly_out_path, f'SMA_gs_1km_annual_{year}.tif'))

    del annual_sma_gs

    ## from here on: indicator calculation -----------------------------------------------------------------------------
    sma_gs = sma_gs.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

    logger.info("indicator calculation")
    # ANNUAL DROUGHT PRESSURE MASK -------------------------------------------------------------------------------------
    sma_ann = sma_gs.mean(dim='time')  # annual seasonally averaged SMA
    ann_drought_pressure = xr.where(sma_ann < -1, sma_ann, np.nan)
    ann_drought_pressure = xr.where(gs_mask == 0, ann_drought_pressure["sma"], np.nan)
    ann_drought_pressure.rio.write_nodata(-999, inplace=True)
    ann_drought_pressure.rio.write_crs("epsg:3035", inplace=True)
    ann_drought_pressure.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    yearly_out_path = os.path.join(out_path, "drought_pressure_mask_gs_1km")
    if not os.path.exists(yearly_out_path):
        os.mkdir(yearly_out_path)
    ann_drought_pressure_masked = ann_drought_pressure.rio.clip(shp_aoi.geometry.values, all_touched=True)
    ann_drought_pressure_masked.rio.to_raster(os.path.join(yearly_out_path, f'drought_pressure_mask_gs_1km_{year}.tif'))

    # ANNUAL DROUGHT PRESSURE OCCURRENCE -------------------------------------------------------------------------
    sma_negatives = xr.where(sma_gs["sma"] < -1, sma_gs["sma"], np.nan)
    drought_pressure_occurrence = sma_negatives.count("time")

    drought_pressure_occurrence = xr.where(ann_drought_pressure<-1, drought_pressure_occurrence, np.nan)
    drought_pressure_occurrence.rio.write_nodata(-999, inplace=True)
    drought_pressure_occurrence.rio.write_crs("epsg:3035", inplace=True)
    drought_pressure_occurrence.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    yearly_out_path = os.path.join(out_path, "drought_pressure_occurrence_gs_1km")
    if not os.path.exists(yearly_out_path):
        os.mkdir(yearly_out_path)
    drought_pressure_occurrence = drought_pressure_occurrence.rio.clip(shp_aoi.geometry.values, all_touched=True)
    drought_pressure_occurrence.rio.to_raster(os.path.join(yearly_out_path, f'drought_pressure_occurrence_gs_1km_{year}.tif'))

    # ANNUAL DROUGHT PRESSURE INTENSITY --------------------------------------------------------------------------------
    drought_pressure_intensity = sma_negatives.mean("time")
    drought_pressure_intensity = xr.where(ann_drought_pressure<-1, drought_pressure_intensity, np.nan)
    drought_pressure_intensity.rio.write_nodata(-999, inplace=True)
    drought_pressure_intensity.rio.write_crs("epsg:3035", inplace=True)
    drought_pressure_intensity.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    yearly_out_path = os.path.join(out_path, "drought_pressure_intensity_gs_1km")
    if not os.path.exists(yearly_out_path):
        os.mkdir(yearly_out_path)
    drought_pressure_intensity = drought_pressure_intensity.rio.clip(shp_aoi.geometry.values, all_touched=True)
    drought_pressure_intensity.rio.to_raster(os.path.join(yearly_out_path, f'drought_pressure_intensity_gs_1km_{year}.tif'))


    return None


if __name__ == "__main__":
    # add log file
    logger.add("99_logfiles\logfile_SMA.log")

    base1 = 2000
    base2 = 2022
    base_path = Path(r'L:\f02_data\drought_indicator')
    aoi_coords = dict(EU=(2500000, 750000, 7500000, 5500000))

    for aoi_name,aoi_bbox in aoi_coords.items():
        for year in range(base1, base2):
            calc_sma_gs_indicators(aoi_name=aoi_name, aoi_coords=aoi_bbox, year=year, base_path=base_path)
