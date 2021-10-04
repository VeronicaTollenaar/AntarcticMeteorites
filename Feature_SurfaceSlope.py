# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd
import tifffile as tiff
import geopandas
from rasterio import features
from affine import Affine
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

# masking and filtering takes long - can be skipped by opening 
# elevation_masked_filtered15.nc directly (cell 4)
#%%
# open elevation data with tiff (because BigTIFF is not supported by rasterio)
elevation_tif = tiff.imread('../Data_raw/REMA_200m_dem_filled.tif')
# set missing data points to nan
elevation_tif = elevation_tif.astype('float')
elevation_tif[elevation_tif==-9999] = np.nan
elevation_tif = elevation_tif.astype('float32')

# assign coordinates (hardcoded)
xul = -2700000.0 + 100
yul = 2300000.0 - 100
res = 200.
(sizey, sizex) = 22500, 27500
x = np.arange(xul,xul+sizex*res, res)
y = np.arange(yul+sizey*-res+res,yul+res, res)[::-1]

# save as DataArray
elevation_DA = xr.DataArray(elevation_tif[:,:],dims=['y','x'],
                  coords={'x': x, 'y': y})
# delete unnecessary variables
del(elevation_tif,res,sizex,sizey,x,xul,y,yul)

#%%
# mask outcrops before filtering
# import outcrop data (shapefile)
outcrops_shp = geopandas.read_file(
    '../Data_raw/Rock_outcrop_medium_res_polygon_v7.1.shx')
# define shape of output
shape_out = np.ones((22500,27500))
# assign value of 1 to all cells within raster
outcrops_mask = features.rasterize(((feature['geometry'], 0) for feature in 
                                    outcrops_shp.iterfeatures()),
                    all_touched = True, out=shape_out, 
                    transform = Affine(200.0, 0.0, -2700000.0,
                                       0.0, -200.0, 2300000.0))
# mask elevation with outcrops_mask
elevation_masked = elevation_DA.where(outcrops_mask==1)
# save as netcdf
#elevation_masked.to_netcdf('demmasked.nc')
# delete unnecessary variables
del(elevation_DA,outcrops_mask,outcrops_shp,shape_out)

#%%
# filter elevation data with averaging filter
# define footprint of averaging filter (circular to avoid sensitivity to the orientation of the axes)
version = 'slope2km'
footprint = np.zeros((11,11)) #number*0.2 = ... km (e.g. 11*0.2 = 2.2 km), number neets to be odd to ensure symmetry
mid = (len(footprint)-1)/2
radius = (len(footprint)-1)/2
for i in range(len(footprint)):
    for j in range(len(footprint)):
        if np.sqrt(((i-mid)**2 + (j-mid)**2)) <= radius:
            footprint[i,j]=1
plt.imshow(footprint)      
#sum(sum(footprint))
# apply filter to elevation data
elevation_filtered = generic_filter(elevation_masked.values,
                                    np.nanmean, footprint = footprint)
# store  in Dataset
elevation_raw = xr.Dataset({
               "elevation_masked_filtered": (("y", "x"), elevation_filtered)},
               coords={"x": elevation_masked.x.values, 
                       "y": elevation_masked.y.values})
# save dataset as nc
#elevation_raw.to_netcdf('Raw_data/elevation_masked_filtered_sizefilter.nc')
# delete unnecessary variables
del(elevation_masked, footprint, mid, radius, elevation_filtered, i, j)
#%%
# if skipping cell 1 - 3: open netcdf
#elevation_raw = xr.open_dataset('Raw_data/elevation_masked_filtered15.nc')

# calculate maximum slope
slope_x = elevation_raw.differentiate('x')
slope_y = elevation_raw.differentiate('y')
slope_max = np.sqrt(slope_x.elevation_masked_filtered**2 
                  + slope_y.elevation_masked_filtered**2)

#generate new dataset with both variables
slope_raw = xr.Dataset({'elevation': elevation_raw.elevation_masked_filtered,
                        'slope_max': slope_max})
# save dataset as nc
#slope_raw.to_netcdf('Raw_data/slope_max.nc')
# delete unnecessary variables
del(slope_x, slope_y, slope_max, elevation_raw)

#%%
# function to extract data
def extractdata(
        data_raw,savename,variablename,file_locs_mets,file_locs_toclassify):
    # import meteorite locations (gridded)
    locs_mets = pd.read_csv('../Data_Locations/'+file_locs_mets+'.csv')
    # select values at meteorite locations
    data_at_mets = data_raw.interp(
        x=locs_mets.x.to_xarray(), 
        y=locs_mets.y.to_xarray())
    # export data at meteorite locations
    data_at_mets_df = data_at_mets.to_dataframe()[
        ['x','y',variablename]]
    data_at_mets_df.to_csv(
        '../Data_Features/'+savename+'_at_mets.csv',
        header=True,index=False)
    
    # import locations to be classified
    locs_toclass = pd.read_csv('../Data_Locations/'+file_locs_toclassify+'.csv')
    # select values at locations to be classified
    data_at_toclass = data_raw.interp(
        x=locs_toclass.x.to_xarray(), 
        y=locs_toclass.y.to_xarray())
    # export where data is NaN
    data_at_toclass_df = data_at_toclass.to_dataframe()[
        ['x','y',variablename]]
    data_at_toclass_nans = data_at_toclass_df[
        np.isnan(data_at_toclass_df[variablename])]
    data_at_toclass_nans.to_csv(
        '../Data_Features/Missing_Values/'+savename+'nans.csv',
        header=True,index=False)
    # export data
    data_at_toclass_nonan = data_at_toclass_df.dropna()
    data_at_toclass_nonan.to_csv(
        '../Data_Features/'+savename+'_at_toclass.csv',
        header=True,index=False)
    print('number of non-nan values at to classify locations (',
          len(locs_toclass),') :', len(data_at_toclass_nonan),'--> ',
          len(locs_toclass)-len(data_at_toclass_nonan), 'nan-values')
    return data_at_mets_df, data_at_toclass_df

#%%
# extract data at locations where meteorites have been found
# extract data at locations that are to be classified
slope_at_mets, slope_at_toclass = extractdata(
    slope_raw,
    version,
    'slope_max',
    'locations_mets',
    'locations_toclass')

#%% 
# extract data at exact finding locations
slope_at_mets, slope_at_toclass = extractdata(
    slope_raw,
    version+'_exactlocs',
    'slope_max',
    'locations_mets_nogrid',
    'locations_toclass')


