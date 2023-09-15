import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
import numpy as np
import pymannkendall
import glob
from datetime import datetime
from loguru import logger

@logger.catch
def extract_trends(base_path):

    variables_dict = {"SMA_annual": "SMA_gs_1km_annual",
                      "drought_pressure_intensity": "drought_pressure_intensity_gs_1km",
                      "drought_pressure_occurrence": "drought_pressure_occurrence_gs_1km"
                      }
    # create a folder for results of trend analysis
    out_path = os.path.join(base_path, "trend_analysis")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # loop through each variable for which a trend analysis is carried out
    for v_name, v_path in variables_dict.items():
        logger.info(f"compute  {v_name}")
        in_path = os.path.join(base_path, v_path)

        filenames = glob.glob(in_path + "/*.tif")

        f0 = filenames[0]
        v_yearly = rio.open_rasterio(f0)
        v_yearly = v_yearly.mean("band")
        v_yearly = v_yearly.assign_coords({"time": pd.to_datetime(f0.split("_")[-1].split(".")[0] + "-12-31")})

        for f in filenames:
            logger.info(f"append  {f}")
            if f != f0:
                sma_tmp = rio.open_rasterio(f)
                sma_tmp = sma_tmp.mean("band")
                sma_tmp = sma_tmp.assign_coords({"time": pd.to_datetime(f.split("_")[-1].split(".")[0]+"-12-31")})

                v_yearly = xr.concat([v_yearly, sma_tmp], "time")

        # identify suitable pixels for trend analysis
        xy_tuples = v_yearly.mean("time").to_series()
        xy_tuples[xy_tuples == -999] = np.nan
        xy_tuples = xy_tuples.dropna().index

        mk_statistics = xr.Dataset({
            "slope": xr.DataArray(
                data=np.nan,
                dims=["y", "x"],
                coords={"y": v_yearly.y.values,
                        "x": v_yearly.x.values}),
            "p_value": xr.DataArray(
                data=np.nan,
                dims=["y", "x"],
                coords={"y": v_yearly.y.values,
                        "x": v_yearly.x.values})
        })

        logger.info("compute Mann-Kendall slope and p-value")
        for i in xy_tuples:
            x_tmp, y_tmp = i[1], i[0]
            ts_tmp = v_yearly.sel(x=x_tmp, y=y_tmp).to_series()
            ts_tmp[ts_tmp == -999] = np.nan
            ts_tmp = ts_tmp.dropna()

            if len(ts_tmp) > 1:
                mk_statistics["slope"].loc[dict(x=x_tmp, y=y_tmp)] = pymannkendall.original_test(ts_tmp).slope
                mk_statistics["p_value"].loc[dict(x=x_tmp, y=y_tmp)] = pymannkendall.original_test(ts_tmp).p
            else:
                mk_statistics["slope"].loc[dict(x=x_tmp, y=y_tmp)] = 0
                mk_statistics["p_value"].loc[dict(x=x_tmp, y=y_tmp)] = 0

        v_out_path = os.path.join(out_path, v_name)
        if not os.path.exists(v_out_path):
            os.mkdir(v_out_path)

        # save a tif file with 2 bands: slope and p_value from Theil-Sen trend analysis
        logger.info("save results")
        mk_statistics = mk_statistics.fillna(-999)
        mk_statistics['slope'].rio.write_nodata(-999, inplace=True)
        mk_statistics['p_value'].rio.write_nodata(-999, inplace=True)

        mk_statistics.rio.write_crs("epsg:3035", inplace=True)
        mk_statistics.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        mk_statistics.transpose('y', 'x').rio.to_raster(os.path.join(v_out_path, "trend_" + v_name + ".tif"), compress='LZW')

    return None


if __name__ == "__main__":

    # add log file
    logger.add("99_logfiles\logfile_SMA_trends.log")
    
    # base_path = Path(r'C:\Users\Zappa\Documents\ETC_DI\drought_indicator\output\SMA_gs_1km_EU')
    base_path = Path(r'L:\f02_data\drought_indicator\output\SMA_gs_1km_EU')

    extract_trends(base_path=base_path)
