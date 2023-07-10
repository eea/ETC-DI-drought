import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from datetime import datetime
import calendar
import itertools


# Calculate growing-season masked SMA
# 0) Calculate SMA zscores first -> CWS_calc_sm_zscore.py
# 1) Reformat SOS and EOS (YYDOY -> DOY) using reclassify_dsos/eos.ipynb
# 2) Load SOS and EOS dates, reformat
# 3) Load SM anomalies, resample to 500m grid, mask outside of growing season
# 4) Save dekadal, growing-season masked SMA files (from earliest SOS date to latest EOS date in study region - more than 1 year, i.e. more than 36 dekads, if SOS in previous year)


def resample_sm(sma_in, src_y, src_x, target_y, target_x, k=1):
    """ Resample SMA from source grid (5 km) to target grid (500 m) using bivariate spline
    """
    # replace NaNs with default SMA value (0)
    data_arr = np.nan_to_num(np.flipud(sma_in), nan=0)  # x and y coords must be monotonically increasing -> flip upside down

    # calculate bivariate spline function on coarse grid
    interp_spline = RectBivariateSpline(src_y[::-1], src_x, data_arr, kx=k, ky=k)

    # apply spatial interpolation on fine grid
    sma_out = np.flipud(interp_spline(target_y[::-1], target_x))

    return sma_out


""" Input:
- 5 km SM anomalies (calculate in CWS_calc_sm_zscore.py)
- 500 m SOS and EOS dates

1) resample SMA to SOS/EOS grid
2) mask SMA outside growing season (keep values between SOS and EOS dates)
3) write SMA-GS to tif-file

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

base_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought')

#coordinates are given xmin, ymin, xmax, ymax, W, S, E, N
# aoi_bbox = dict(west=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2,5500000.0000 ),
#               east=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000,5500000.0000 ))
aoi_bbox = dict(Nwest=(1500000.0000, 900000.0000+(5500000.0000-900000.0000)/2, 1500000.0000+(7400000.0000-1500000.0000)/2, 5500000.0000 ),
                Neast=(1500000.0000+(7400000.0000-1500000.0000)/2, 900000.0000+(5500000.0000-900000.0000)/2, 7400000.0000, 5500000.0000 ),
                Swest=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2, 900000.0000+(5500000.0000-900000.0000)/2 ),                
                Seast=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000, 900000.0000+(5500000.0000-900000.0000)/2))
                  
for aoi_name in list(aoi_bbox):
    aoi_coords = aoi_bbox[aoi_name]
    for year in range(2000, 2022):
        # AOI: subset of Europe, e.g. Iberian peninsula (in order to get everything into memory)
        # (adding 1x spacing to make sure 5 km SMA covers entire AOI)
        spacing = 5000
        # spacing = 0
        xmin = aoi_coords[0] - spacing
        ymin = aoi_coords[1] - spacing
        xmax = aoi_coords[2] + spacing
        ymax = aoi_coords[3] + spacing

        sos_path = os.path.join(base_path, 'SOS')
        eos_path = os.path.join(base_path, 'EOS')
        sma_path = os.path.join(base_path, 'JRC_SMA_custom')
        # out_path = os.path.join(base_path, f'JRC_SMA_GS_{aoi_name}')
        out_path = os.path.join(base_path, f'MRVPP_workflow_results_{aoi_name}')
        sm_var = 'sma'
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        ################################################## SOS & EOS ######################################################
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

        ######################################### SM in growing season (2001 onwards) #########################################
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

            # resample and interpolate each timestamp
            sma500m = np.empty((len(ds_time), len(target_y), len(target_x)))
            for tsi in range(len(ds_time)):
                sma_tsi = resample_sm(sma5k.isel(band=tsi)[sm_var].values, src_y, src_x, target_y, target_x)
                sma500m[tsi] = sma_tsi
            # store in new xarray dataset
            del sma5k, sma_tsi
            sma_ds = xr.Dataset(
                                data_vars=dict(sma=(['time', 'y', 'x'], sma500m)),
                                coords=dict(
                                    time=ds_time,
                                    y=target_y,
                                    x=target_x,
                                ),
                                attrs=dict(description="JRC SMA resampled to 500m grid using bivariate spline interpolation."),
                            )

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
            sma500m = np.empty((len(ds_time), len(target_y), len(target_x)))
            for tsi in range(len(ds_time)):
                sma_tsi = resample_sm(sma5k.isel(band=tsi)['sma'].values, src_y, src_x, target_y, target_x)
                sma500m[tsi] = sma_tsi
            # store in new xarray dataset
            del sma5k, sma_tsi
            sma_ds = xr.Dataset(
                data_vars=dict(sma=(['time', 'y', 'x'], sma500m)),
                coords=dict(
                    time=ds_time,
                    y=target_y,
                    x=target_x,
                ),
                attrs=dict(description="JRC SMA resampled to 500m grid using bivariate spline interpolation."),
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
        del sos_dekads, eos_dekads, gs_mask
        
        # remove timestamps where data is all NaN (all bands before/after earliest/latest SOS/EOS date)
        sma_gs = sma_gs.dropna(dim='time', how='all')

        # Removed 2023-04-03: save SMA-GS to tiff files (to avoid memory error)
        # save sma_gs to tiff (each file containing dekadal values from one year or more, depending on earliest SOS date)
        # sma_gs = sma_gs.fillna(-999)
        # dataset with coords (time, y, x) to dataset with coords (y, x) and timestamps stored as separate bands
        # needed to store timestamp info in dekadal tiff files
        # will also show timestamps as band names in QGIS
        # bands = [pd.to_datetime(str(dt)).strftime('%Y-%m-%d') for dt in sma_gs.time.values]
        # sma_gs_dt = xr.Dataset()
        # for bi, band in enumerate(bands):
        #     sma_gs_dt[band] = xr.DataArray(sma_gs.isel(time=bi)['sma'].values, coords={'y': sma_gs.y, 'x': sma_gs.x})
        #     sma_gs_dt[band].rio.write_nodata(-999, inplace=True)
        # sma_gs_dt.rio.write_crs("epsg:3035", inplace=True)
        # sma_gs_dt.rio.to_raster(os.path.join(out_path, f'sma_gs_{year}.tif'), compress='LZW')

        ### from here onwards: indicator calculation (and save to tif)
        # redefine bounding box to match new 500 m resolution and avoid AOI overlaps
        spacing = 500
        xmin = aoi_coords[0] - spacing
        ymin = aoi_coords[1] - spacing
        xmax = aoi_coords[2] + spacing
        ymax = aoi_coords[3] + spacing
        sma_gs = sma_gs.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
        
        # MONTHLY DROUGHT HAZARD - uncomment if needed
        # sma_mon = sma_gs.resample(time='MS').mean()
        # mon_drought_hazard = xr.where(sma_mon < 0, sma_mon, np.nan) # calculated but not written out

        # ANNUAL DROUGHT PRESSURE (average over growing season)
        sma_ann = sma_gs.mean(dim='time')   # annual seasonally averaged SMA
        ann_drought_pressure = xr.where(sma_ann < -1, sma_ann, np.nan)

        # ANNUAL DROUGHT PRESSURE INTENSITY
        sma_mon_min = sma_gs.resample(time='MS').min()
        sma_mon_min_neg = xr.where(sma_mon_min < -1, sma_mon_min, np.nan)
        sma_ann_dpi = sma_mon_min_neg.mean(dim='time')

        # free up space
        del sma_gs

        ################################################ SM RESULTS TO TIFF ######################################################
        results = dict(sma_gs_avg=sma_ann, sm_ann_dp=ann_drought_pressure, sm_ann_dpi=sma_ann_dpi)
        for res in list(results):
            v = 'sma'
            results[res] = results[res].fillna(-999)
            results[res][v].rio.write_nodata(-999, inplace=True)
            results[res].rio.write_crs("epsg:3035", inplace=True)
            results[res][v].rio.to_raster(os.path.join(out_path, f'{res}_{year}.tif'))
        # free up space
        del sma_ann, sma_ann_dpi, ann_drought_pressure