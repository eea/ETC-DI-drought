import os
from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS


def merger(src, flist, fname, out_path):
    mosaic, out_trans = merge(flist)

    # Copy the metadata
    out_meta = src.meta.copy()
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +type=crs"})

    out_fp = os.path.join(out_path, fname[0])
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.crs = CRS.from_epsg(3035)
        dest.write(mosaic)


""" Mosaic tiffs together

Parameters:
    aois: list of str
        e.g. ['Europe-west', 'Europe-east'] - as defined in other python scripts
    start_year, end_year: int
        first and last year with data
    base_path: path object
        Path to results folder (MRVPP_workflow_results_<aoi>)
    varis: list of str
        list of variables. All available tiffs per variable and year (from different AOIs) will be merged.
        Make sure that the raster/variable names are unique!
"""
aois = ['Nwest', 'Neast', 'Swest', 'Seast']
start_year = 2000
end_year = 2021 # last year for which we have data
base_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought')
folder_name = 'MRVPP_workflow_results'  # Folder name (without '_aoi') that contains the AOI-files that should be merged. e.g. 'MRVPP_workflow_results' for SMA- and LINT-based and LTA drought indicators, 'LINT_anom' for lint zscores, etc.
# drought indicator variables
varis = ['sma_gs_avg_', 'sm_ann_dp_', 'sm_ann_dpi_', 'lint_mr_di', 'sma_lta_dp.', 'sma_lta_dpi.', 'lint_lta_di']

aoi_str = '_'.join(aois)

out_path = os.path.join(base_path, f'{folder_name}_mosaic')
if not os.path.exists(out_path):
    os.mkdir(out_path)

for vari in varis:
    if 'lta' in vari:
        flist = []
        for aoi in aois:
            path = os.path.join(base_path, f'{folder_name}_{aoi}')
            fname = [f for f in os.listdir(path) if vari in f]
            print(fname)
            src = rasterio.open(os.path.join(path, fname[0]))
            flist.append(src)
        merger(src, flist, fname, out_path)
    else:
        for year in range(start_year, end_year+1):
            flist = []
            for aoi in aois:
                path = os.path.join(base_path, f'{folder_name}_{aoi}')
                fname = [f for f in os.listdir(path) if str(year) in f and vari in f]
                print(fname)
                src = rasterio.open(os.path.join(path, fname[0]))
                flist.append(src)
            merger(src, flist, fname, out_path)
