# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd
import rasterio
import geopandas
from rasterio import features
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

# calculating distances to outcrops takes long - can be skipped by opening 
# distance_to_outcrops.nc directly (cell 5)

#%%
# draw increasing buffers around polygons of outcrops and rasterize these,
# where a value is assigned approximating the distance to the nearest outcrop

# open data outcrops (polygons)
outcrops_polygon = geopandas.read_file(
    '../Data_raw/Rock_outcrop_medium_res_polygon_v7.1.shx')

# open data raster to project polygons on
raster_transform = rasterio.open(
    'netcdf:../Data_raw/BedMachineAntarctica_2020-07-15_v02.nc:bed')
for_coordinates = xr.open_dataset(
    '../Data_raw/BedMachineAntarctica_2020-07-15_v02.nc')
#%%
# define buffer sizes
buffer_sizes = np.around(np.logspace(np.log10(400),np.log10(1500),750))
plt.scatter(np.linspace(1,len(buffer_sizes),len(buffer_sizes)),buffer_sizes)
plt.title('increasing buffer sizes (m)')
print('max distance is',sum(buffer_sizes)/1000)
# calculate value to assign to each additional buffer
value_to_assign = np.zeros(len(buffer_sizes))
for k in range(len(value_to_assign)):
    value_to_assign[k] = sum(buffer_sizes[:k])+buffer_sizes[k]/2
del(k)

# define function to draw buffer
def polyf(poly,buffer_size):
    return poly.buffer(buffer_size)
#%%
# rasterize data outcrops (polygons)
# define shape of output
shape_out = np.empty(np.shape(raster_transform))
shape_out[:] = np.nan
# assign value of 0 to all cells within raster
iteration_0 = features.rasterize(((feature['geometry'], 0) for feature 
                         in outcrops_polygon.iterfeatures()),
                         all_touched = False, 
                         out = shape_out, 
                         transform = raster_transform.transform)

# iterations to rasterize buffers
crs = {'init': 'epsg:3031'} # set crs
iteration_previous = iteration_0

for i in range(len(buffer_sizes)):
    outcrops_polygon['additional_buffer']=outcrops_polygon.geometry.apply(polyf,
                                                    buffer_size=buffer_sizes[i])
    # make a union of the separate polygons
    additional_buffer_union = cascaded_union(outcrops_polygon[
                              'additional_buffer'].values)
    # store the union of polygons in a dataframe
    try: # if multiple polygons
        additional_buffer_polygon = pd.DataFrame(list(additional_buffer_union),
                                                 columns=['geometry'])
    except TypeError: # if single polygon
        additional_buffer_polygon = pd.DataFrame([additional_buffer_union],
                                                 columns =['geometry'])
    # create a geo data frame of the union of polygons
    additional_buffer_gdf = geopandas.GeoDataFrame(additional_buffer_polygon,
                                 crs=crs,
                                 geometry=additional_buffer_polygon.geometry)
    # create an array of the rasterized polygons of previous iteration
    iteration_previous_array = np.array(iteration_previous)
    # rasterize the current geo data frame of union of polygons
    iteration_current = features.rasterize(((feature['geometry'],
                                 value_to_assign[i]) for feature 
                                 in additional_buffer_gdf.iterfeatures()),
                                 all_touched=False, 
                                 out=shape_out,
                                 transform=raster_transform.transform)
    # overwrite the values of current rasterized iteration with the values of
    # previous rasterized iterations
    iteration_current[~np.isnan(iteration_previous_array)]=iteration_previous_array[
                                 ~np.isnan(iteration_previous_array)]
    # set the current polygon as the polygon for next iteration
    del(outcrops_polygon)
    outcrops_polygon = additional_buffer_gdf
    # set the current iteration as  previous iteration
    del(iteration_previous)
    iteration_previous = iteration_current
    del(iteration_current,additional_buffer_union,
        additional_buffer_polygon,additional_buffer_gdf)
    print('distance of '+ str(np.round(value_to_assign[i]/1000,2)) +' km completed')
#%%
# store in DataArray
distanceoutcrops_DA = xr.DataArray(iteration_previous,
                                   coords=(for_coordinates.y, for_coordinates.x))
# create dataset from DataArray
distanceoutcrops_raw = distanceoutcrops_DA.to_dataset(name='distanceoutcrops')
# export as netcdf
distanceoutcrops_raw.to_netcdf('../Data_raw/distance_to_outcrops.nc')
# delete unnecessary variables
del(buffer_sizes, crs, distanceoutcrops_DA, for_coordinates, i,
    iteration_0, iteration_previous, iteration_previous_array,
    outcrops_polygon, raster_transform, shape_out, value_to_assign)
#%%
# if skipping cell 1 - 4: open netcdf
distanceoutcrops_raw = xr.open_dataset('../Data_raw/distance_to_outcrops.nc')
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
    return data_at_mets_df, data_at_toclass_nonan

#%%
# extract data at locations where meteorites have been found
# extract data at locations that are to be classified
distanceoutcrops_at_mets, distanceoutcrops_at_toclass = extractdata(
    distanceoutcrops_raw,
    'distanceoutcrops',
    'distanceoutcrops',
    'locations_mets',
    'locations_toclass')


