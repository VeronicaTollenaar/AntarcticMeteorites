# import packages
import numpy as np
import xarray as xr
import os
import rasterio
import geopandas
from rasterio import features

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)



#%%
## Open shapefile and transform to raster
# Shapefile = BIAs of quantarctica with 1-km buffer 
# Raster resolution = 450 m (as velocity data)

# extract transform/Affine information from Surface Velocity data
dataset = rasterio.open('netcdf:../Data_raw/antarctic_ice_vel_phase_map_v01.nc:VX')
dataset.transform
# import shapefile
polygons = geopandas.read_file('../Data_raw/bias_above200m1kmbuff_expanded_dissolved.shx')

# define shape of output
out = np.zeros(np.shape(dataset))
# assign value of 1 to all cells within raster
polygons_rasterized = features.rasterize(((feature['geometry'], 1) for feature in polygons.iterfeatures()),
                       all_touched = True, out=out, transform = dataset.transform)

#%%
# check the created mask
print(np.shape(polygons_rasterized))
print('total of pixels in polygons: ',np.sum(polygons_rasterized))

# import coordinates
coords_vel = xr.open_dataset('../Data_raw/antarctic_ice_vel_phase_map_v01.nc')

# save as DataArray
maskDA = xr.DataArray(polygons_rasterized, coords=(coords_vel.y, coords_vel.x))


#%%
# Save as csv with coordinates (similar to positive observations)
xys = maskDA.to_dataframe('bias') #bias stands for blue ice areas
xys = xys[xys.bias==1]

xys.to_csv('../Data_Locations/locations_toclass.csv')

#%%
# Save with indexno for calculations in Earth Engine
xys_ee = xys.copy()
xys_ee['indexno'] = np.linspace(0,len(xys_ee)-1,len(xys_ee))
xys_ee.to_csv('../Data_Locations/locations_toclass_EE.csv')





