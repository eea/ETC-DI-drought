import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np
from scipy.interpolate import RectBivariateSpline
from datetime import datetime
import matplotlib.pyplot as plt
import calendar

# Calculate drought indicators using SMA and LINT
# 0) Calculate LINT zscores first -> CWS_calc_lint_zscore.py
# 1) Reformat SOS and EOS (YYDOY -> DOY) using reclassify_dsos/eos.ipynb
# 2) Load SOS and EOS dates, reformat
# 3) Load SM anomalies, resample to 500m grid, mask outside of growing season
# 4) Load LINT anomalies
# 5) Calculate SM- and LINT-based indicators

# Steps 2-5 are done per year. Long-term average indicators are calculated in a separate script (CWS_lta_indicators.py)



def resample_sm(sma_in, src_y, src_x, target_y, target_x, k=1):
    # replace NaNs with default SMA value (0)
    data_arr = np.nan_to_num(np.flipud(sma_in), nan=0)  # x and y coords must be monotonically increasing -> flip upside down

    # calculate bivariate spline function on coarse grid
    interp_spline = RectBivariateSpline(src_y[::-1], src_x, data_arr, kx=k, ky=k)

    # apply spatial interpolation on fine grid
    sma_out = np.flipud(interp_spline(target_y[::-1], target_x))
    sma_out[sma_out == 0] = np.nan

    return sma_out

def calc_indicators(year):
    # AOI: subset of Europe, e.g. Iberian peninsula (in order to get everything into memory)
    #coordinates are given xmin, ymin, xmax, ymax, W, S, E, N
    # (adding 1x spacing to make sure SMA covers entire AOI)
    # EEA_Area=( 2555000, 1350000 , 7405000,5500000 )
    # EEA_Area_west=( 2555000, 1350000 , 7405000/2,5500000 ),
    #EEA_Area_east=( 7405000/2, 1350000 , 7405000,5500000 ))
    aoi = 'east'
    aoi_coords = dict(Iberia=(2630000, 1580000, 3770000, 2460000),
                      CentEur=(4000000, 2300000, 5000000, 3100000),
                      EEA_Area=( 1499900.0000, 900000.0000, 7399900.0000,5500000.0000 ),
                      west=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2,5500000.0000 ),
                      east=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000,5500000.0000 ))
   
    spacing = 0
    xmin = aoi_coords[aoi][0] - spacing
    ymin = aoi_coords[aoi][1] - spacing
    xmax = aoi_coords[aoi][2] + spacing
    ymax = aoi_coords[aoi][3] + spacing



    base_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought')
    sos_path = os.path.join(base_path, 'SOS')
    eos_path = os.path.join(base_path, 'EOS')
    # lint_path = os.path.join(base_path, f'LINT_anom_{aoi}')
    sma_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought\SMA_nc')
    out_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought', f'GSSMA_{aoi}')
    if not os.path.exists(out_path):
        os.mkdir(out_path)



    ################################################## SOS & EOS ######################################################
    # load SOS dates of desired year into xarray
    tif_list = [f for f in os.listdir(sos_path) if str(year) in f]
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[20:24]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs contained in the given folder
    sos_ds = xr.concat([xr.open_rasterio(os.path.join(sos_path, i)) for i in tif_list], dim=time_var).sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
    # Covert our xarray.DataArray into a xarray.Dataset
    sos_ds = sos_ds.to_dataset('band')
    # Rename the variable to a more useful name
    sos_ds = sos_ds.rename({1: 'dSOS'})
    # Subtract 3 months from SOS
    sos_ds = sos_ds - 90

    # load EOS dates of desired year into xarray
    tif_list = [f for f in os.listdir(eos_path) if str(year) in f]
    # Create variable used for time axis
    time_var = xr.Variable('time', pd.to_datetime([f'{fname[20:24]}-01-01' for fname in tif_list]))
    # Load in and concatenate all individual GeoTIFFs contained in the given folder
    eos_ds = xr.concat([xr.open_rasterio(os.path.join(eos_path, i)) for i in tif_list], dim=time_var).sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
    # Covert our xarray.DataArray into a xarray.Dataset
    eos_ds = eos_ds.to_dataset('band')
    # Rename the variable to a more useful name
    eos_ds = eos_ds.rename({1: 'dEOS'})
    # replace 0s by year-365 but keep mask of EOS=0 pixels (to be applied later)
    eos_undef = xr.where(eos_ds == 0, 1, 0)
    eos_ds = xr.where(eos_ds == 0, 365 + calendar.isleap(year), eos_ds).astype(int)
    # if EOS falls into next year, set to Dec. 31st
    eos_ds = xr.where(eos_ds > 365 + calendar.isleap(year), 365 + calendar.isleap(year), eos_ds).astype(int)

    ######################################### SM in growing season (2001 onwards) #########################################
    if year != 2000:
        # load soil moisture anomalies into xarray
        sma5k = xr.open_mfdataset(os.path.join(sma_path, '*.nc')).sel(time=slice(datetime(year-1, 1, 1), datetime(year, 12, 31))).sel(lon=slice(xmin, xmax), lat=slice(ymax, ymin))
        src_x = sma5k.lon
        src_y = sma5k.lat
        target_x = sos_ds.x
        target_y = sos_ds.y
        ds_time = sma5k.time

        # resample and interpolate each timestamp
        sma500m = np.empty((len(ds_time), len(target_y), len(target_x)))
        for tsi in range(len(ds_time)):
            sma_tsi = resample_sm(sma5k.isel(time=tsi)['smian'].values, src_y, src_x, target_y, target_x)
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

        # check result of SMA resampling
        # fig, ax = plt.subplots(2, 1)
        # ax[0].imshow(sma5k['smian'].sel(time=datetime(2021, 7, 1)))
        # ax[1].imshow(sma_ds['sma'].sel(time=datetime(2021, 7, 1)))
        # plt.show()

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
        # shift SOS dates to year-1 (to have only positive values, ranging from 1-365*2 for 2 years of data)
        sos_doys = sos_ds['dSOS'].values + 365 + calendar.isleap(year-1)
        # convert SOS dates from DOYs to dekads using doys_lut
        sos_dekads = doys_lut[sos_doys-1]
        del sos_doys, sos_ds

        # shift also EOS by one year to match format of SOS and SMA
        eos_doys = eos_ds['dEOS'].values + 365 + calendar.isleap(year-1)
        eos_dekads = doys_lut[eos_doys-1]
        del eos_doys, eos_ds

        # sos_dekads and eos_dekads can now be used for indexing and masking sma_ds
        # (sos_dekad 1 corresponds to index 0, i.e. first timestamp in sma_ds, etc.)
        dekads_arr = np.repeat(np.repeat(np.arange(1, 73)[:, np.newaxis], len(target_y), axis=1)[:, :, np.newaxis], len(target_x), axis=2)
        sma_gs = xr.where((dekads_arr >= sos_dekads) & (dekads_arr <= eos_dekads), sma_ds, np.nan)
        del sma_ds
        # apply mask of undefined EOS
        eos_undef = np.repeat(eos_undef['dEOS'].values[0][np.newaxis, :, :], len(sma_gs['time']), axis=0)
        sma_gs = xr.where(eos_undef == 0, sma_gs, np.nan)

    ################################ SM in growing season (exception for year 2000) ######################################
    else:
        # no SMA data before 2000 -> slightly different approach: set all SOS from 1999 to 1.1.2000
        # load soil moisture anomalies into xarray
        sma5k = xr.open_mfdataset(os.path.join(sma_path, '*.nc')).sel(time=slice(datetime(year, 1, 1), datetime(year, 12, 31))).sel(lon=slice(xmin, xmax), lat=slice(ymax, ymin))
        src_x = sma5k.lon
        src_y = sma5k.lat
        target_x = sos_ds.x
        target_y = sos_ds.y
        ds_time = sma5k.time

        # resample and interpolate each timestamp
        sma500m = np.empty((len(ds_time), len(target_y), len(target_x)))
        for tsi in range(len(ds_time)):
            sma_tsi = resample_sm(sma5k.isel(time=tsi)['smian'].values, src_y, src_x, target_y, target_x)
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
        # set negative SOS dates to 1st day of year
        sos_doys = sos_ds['dSOS'].values
        sos_doys[sos_doys <= 0] = 1
        # convert SOS dates from DOYs to dekads using doys_lut
        sos_dekads = doys_lut[sos_doys - 1]
        del sos_doys, sos_ds

        # set potential negative EOS dates to 1st day of year
        eos_doys = eos_ds['dEOS'].values
        eos_doys[eos_doys <= 0] = 1
        eos_dekads = doys_lut[eos_doys - 1]
        del eos_doys, eos_ds

        # sos_dekads and eos_dekads can now be used for indexing and masking sma_ds
        # (sos_dekad 1 corresponds to index 0, i.e. first timestamp in sma_ds, etc.)
        dekads_arr = np.repeat(np.repeat(np.arange(1, 37)[:, np.newaxis], len(target_y), axis=1)[:, :, np.newaxis],
                               len(target_x), axis=2)
        sma_gs = xr.where((dekads_arr >= sos_dekads) & (dekads_arr <= eos_dekads), sma_ds, np.nan)
        del sma_ds
        # apply mask of undefined EOS
        eos_undef = np.repeat(eos_undef['dEOS'].values[0][np.newaxis, :, :], len(sma_gs['time']), axis=0)
        sma_gs = xr.where(eos_undef == 0, sma_gs, np.nan)

    # free up space
    del sos_dekads, eos_dekads, eos_undef

    ################################################## SM INDICATORS ######################################################
    # MONTHLY DROUGHT HAZARD
    # sma_mon = sma_gs.resample(time='MS').mean()
    # mon_drought_hazard = xr.where(sma_mon < 0, sma_mon, np.nan) # calculated but not written out

    # ANNUAL DROUGHT PRESSURE (average over growing season)
    sma_ann = sma_gs.mean(dim='time')   # annual seasonally averaged SMA
    ann_drought_pressure = xr.where(sma_ann < 0, sma_ann, np.nan)

    # ANNUAL DROUGHT PRESSURE INTENSITY
    # sma_mon_min = sma_gs.resample(time='MS').min()
    # sma_mon_min_neg = xr.where(sma_mon_min < 0, sma_mon_min, np.nan)
    # sma_ann_dpi = sma_mon_min_neg.mean(dim='time')

    # free up space
    del sma_gs

    ################################################ SM RESULTS TO TIFF ######################################################
    # results = dict(sma_gs_avg=sma_ann, sm_ann_dp=ann_drought_pressure, sm_ann_dpi=sma_ann_dpi)
    results = dict(sm_ann_dp=ann_drought_pressure)
    for res in list(results):
        v = 'sma'
        results[res].rio.write_crs("epsg:3035", inplace=True)
        results[res][v].rio.to_raster(os.path.join(out_path, f'{res}_{year}.tif'))
    # free up space
    # del sma_ann, sma_ann_dpi
    # del sma_ann

    # ################################################ MR-VPP INDICATORS ####################################################
    # # load LINT into xarray
    # tif_list = [f for f in os.listdir(lint_path) if str(year) in f]
    # # Create variable used for time axis
    # time_var = xr.Variable('time', pd.to_datetime([f'{fname[10:14]}-01-01' for fname in tif_list]))
    # # Load in and concatenate all individual GeoTIFFs
    # lint_da = xr.concat([xr.open_rasterio(os.path.join(lint_path, i)) for i in tif_list], dim=time_var).sel(y=slice(ymax, ymin), x=slice(xmin, xmax))
    # # Covert our xarray.DataArray into a xarray.Dataset
    # lint_ds = lint_da.to_dataset('band')
    # # Rename the variable to a more useful name
    # lint_ds = lint_ds.rename({1: 'linta'})

    # # ANNUAL MEDIUM RES DROUGHT IMPACT
    # lint_dp = xr.where(ann_drought_pressure['sma'] < 0, lint_ds, np.nan)
    # del lint_ds
    # # above line changes order of dimensions -> known xarray issue. reorder back in next step
    # lint_dp['linta'] = lint_dp['linta'].transpose('time', 'y', 'x')
    # ann_mr_drought_impact = xr.where(lint_dp < -0.5, lint_dp, np.nan)
    # del lint_dp
    
    # ################################################ RESULTS TO TIFF ######################################################
    # results = dict(lint_mr_di=ann_mr_drought_impact)
    # for res in list(results):
    #     v = 'linta'
    #     results[res].rio.write_crs("epsg:3035", inplace=True)
    #     results[res][v].rio.to_raster(os.path.join(out_path, f'{res}_{year}.tif'))


if __name__ == "__main__":
    for year in range(2000, 2022):
        calc_indicators(year=year)
    print('Done.')
