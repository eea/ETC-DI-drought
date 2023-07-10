import os
from pathlib import Path
import xarray as xr
import rioxarray as rio
import pandas as pd
from datetime import datetime


print ("PART00 REading libs")

# Calculate MR-VPP LINT anomalies
# 1) Load all yearly LINT files
# 2) Calculate long-term average and standard deviation over defined base
# 3) Calculate zscores as (annual_LINT - lta_LINT) / stdev_LINT
# 4) Write yearly LINT anomaly files


""" Calculate LINT anomalies: LINTanom = (LINT - LTA) / stdev1
LINTanom...LINT anomaly, LTA...long-term average, stdev...standard deviation

aoi_name...area of interest (must be defined in aoi_coords). Necessary in order to avoid memory error
aoi_coords...bounding box coordinates of AOI
base_path: path object (path to folder that contains LINT folder)
base1, base2...start and end of baseline period

"""




start_year = 2000
end_year = 2022 # last year for which we have data
base_path = Path(r'L:\f02_data\drought_indicator\input')
mask = r"L:\f02_data\ref_data\terrestrial_europe_mask\admin_mask_europe.shp"
output_path_mask = r'L:\f02_data\drought_indicator\input\LINT_EUROPE'

base1 = 2001
base2 = 2020

# aoi_bbox = dict(west=(1500000.0000, 900000.0000, 1500000.0000+(7400000.0000-1500000.0000)/2,5500000.0000 ),
#               east=(1500000.0000+(7400000.0000-1500000.0000)/2,  900000.0000, 7400000.0000,5500000.0000 ))


###############input data update
print ("PART01 cleaning input data")
##clip raster to Europe:
######"gdalwarp -overwrite -s_srs IGNF:ETRS89LAEA -t_srs IGNF:ETRS89LAEA 
        #-of GTiff -cutline L:/f02_data/ref_data/terrestrial_europe_mask/admin_mask_europe.shp
        #-cl admin_mask_europe -dstnodata -999.0 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 
        #L:/drought_indicator/LINT/LINT_mosaic_2000_s1_cog.tif L:/drought_indicator/tmp/mask_output.tif"

def clip_raster(clip_feature, in_raster, out_raster):
    """
    This function will create and execute a GDAL Warp command with the given inputs
    """
    
    print("Clipping raster...")
    
    # creates the gdal warp command
    command = f"gdalwarp -overwrite -s_srs EPSG:3035 -t_srs EPSG:3035 -of GTiff -cutline  {clip_feature} -cl admin_mask_europe -dstnodata -999.0 -co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9  {in_raster} {out_raster}"

    # runs the command
    os.system(command)



##loop over raster inf folder "LINT":
lint_path = os.path.join(base_path, 'LINT')
for filename in os.listdir(lint_path,):
    f = os.path.join(lint_path, filename)

    # checking if it is a file
    if os.path.isfile(f):
        #print(f)
        if f.endswith(".tif"):
            print(f)
            inraster = f
            outraster = output_path_mask+"\\euorope_" + filename
            print (outraster)
            clip_raster(mask, inraster, outraster)





       

print ("DONE----------------")