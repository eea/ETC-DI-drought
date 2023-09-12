# Preprocessing of GDMP 1km and 300m resolution
Preprocessing is done with QGIS batch processing. Here you can fine the config files for the different steps.
Replace `<base folder>` with correct base path in your system.
You may need to adapt the path to your system.

## How to use the config files
1. Open QGIS
2. load all layers to be processed
3. select the appropriate Processing tool
4. Select `Batch processing`
5. Click on the folder icon to load the config file

## GDMP 1km resolution
1. transform to physical values `qgis_get_phys_val_1000m.json` (QGIS tool: Raster calculation)
2. clip with EU mask `qgis_preprocessing_1000m.json` (QGIS tool: Clip raster by mask layer)

## GDMP 300m resolution
1. transform to physical values `qgis_get_phys_val_300m.json` (QGIS tool: Raster calculation)
2. clip with EU mask, resample to 1km with bilinear resampling method `qgis_preprocessing_300m_to_1km.json` (QGIS tool: Clip raster by mask layer)