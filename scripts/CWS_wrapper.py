from pathlib import Path
from CWS_calc_sm_zscore import calculate_SM_anomalies
from CWS_calc_lint_zscore import calculate_lint_anomalies
from CWS_sma_gs import calc_sma_gs
from CWS_sma_indicators import calc_sm_indicators
from CWS_lint_indicators import calc_lint_indicators
from CWS_lta_indicators import calc_LTA_indicators
from CWS_merge_tiffs import tiff_mosaic
from datetime import datetime

# Examples of how to run scripts and in which order
# DONT RUN wrapper - needs restructuring of scripts


# settings
aois = ['west', 'east']
start_year = 2000
end_year = 2021 # last year for which we have data
base_path = Path(r'S:\Common workspace\ETC_DI\AP22_Drought')

# bounding box coordinates given in (xmin, ymin, xmax, ymax)
aoi_coords = dict(west=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2,5500000.0000 ),
                  east=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000,5500000.0000 ))


# calculate SM anomalies - run only once, not for several AOIs!
print(f'{datetime.now()} Calculate SM anomalies...')
calculate_SM_anomalies(base_path, start_year, end_year, base1=2000, base2=2015)

for aoi_name in aois:
    aoi_bbox = aoi_coords[aoi_name]
    # calculate LINT anomalies
    # print(f'{datetime.now()} [{aoi_name}] Calculate LINT anomalies...')
    # calculate_lint_anomalies(aoi_name, aoi_bbox, base_path, start_year, end_year, base1=2000, base2=2015)

    for year in range(start_year, end_year+1):
        # mask SMA outside growing season
        print(f'{datetime.now()} [{aoi_name}, {year}] Mask SM anomalies outside growing season...')
        calc_sma_gs(aoi_name=aoi_name, aoi_coords=aoi_bbox, year=year, base_path=base_path)
    #     # calculate SM-based drought indicators
    #     print(f'{datetime.now()} [{aoi_name}, {year}] Calculate SM indicators...')
    #     calc_sm_indicators(aoi_name=aoi_name, aoi_coords=aoi_bbox, year=year, base_path=base_path)
    #     # calculate SM- and LINT-based drought indicators
    #     print(f'{datetime.now()} [{aoi_name}, {year}] Calculate LINT indicators...')
    #     calc_lint_indicators(aoi_name=aoi_name, aoi_coords=aoi_bbox, year=year, base_path=base_path)

    # # calculate long-term average drought indicators
    # print(f'{datetime.now()} [{aoi_name}] Calculate long-term average indicators...')
    # calc_LTA_indicators(aoi_name, base_path=base_path)

# as soon as indicators for all AOIs (e.g. Europe-west, Europe-east) have been calculated: mosaic results together
print('{datetime.now()} Perform tif mosaicking...')
tiff_mosaic(aois, start_year, end_year, base_path, varis=['sma_gs_avg_'])
# tiff_mosaic(aois, start_year, end_year, base_path, varis=['sma_gs_avg_', 'sm_ann_dp_', 'sm_ann_dpi_', 'lint_mr_di',
#                                                           'sma_lta_dp.', 'sma_lta_dpi.', 'lint_lta_di'])

print('Done.')
