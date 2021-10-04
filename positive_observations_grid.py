# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# Import coordinates of grid
coords_vel = xr.open_dataset('../Data_raw/antarctic_ice_vel_phase_map_v01.nc')

#%%
# Import meteorite finding locations (meteorite_locations)
met_locs = pd.read_csv('../Data_raw/meteorite_locations_raw.csv')

#%%
# reproject on grid by finding nearest neighbours
met_locs_reproj = coords_vel.sel(x=met_locs.new_x.to_xarray(), 
                                 y=met_locs.new_y.to_xarray(),
                                 method='nearest')

# dataset as pandas dataframe
met_locs_reproj_df = met_locs_reproj.to_dataframe()[['x','y']]

#%%
# add abreviations of field sites ('abbrevs') to meteorite locations
met_locs_reproj_df['abbrevs']=met_locs.abbrevs.values

#%%
# count duplicates and create new array with unique coordinates, fieldsite and number of meteorites per gridcell ('counts')
met_locs_reproj_gr = met_locs_reproj_df.groupby(['x','y','abbrevs']).size().reset_index(name='counts')
# drop values with identical coordinates but different field sites
met_locs_reproj_gr = met_locs_reproj_gr.drop(met_locs_reproj_gr[met_locs_reproj_gr.duplicated(['x','y'])].index)

# save as csv (with and without field sites)
met_locs_reproj_gr[['x','y','counts']].to_csv('../Data_Locations/locations_mets.csv',index=False)
met_locs_reproj_gr.to_csv('../Data_Locations/locations_mets_abbrevs.csv',index=False)

#%%
# Save with indexno for calculations in Earth Engine
# reprojected observations
met_locs_reproj_ee = met_locs_reproj_df.copy()
met_locs_reproj_ee['indexno'] = np.linspace(0,
                                            len(met_locs_reproj_ee)-1,
                                            len(met_locs_reproj_ee))
met_locs_reproj_ee.to_csv('../Data_Locations/locations_mets_EE.csv')

# original observations
met_locs_ee = met_locs[['new_x','new_y']].copy()
met_locs_ee['indexno'] = np.linspace(0,
                                     len(met_locs_ee)-1,
                                     len(met_locs_ee))
met_locs_reproj_ee.to_csv('../Data_Locations/locations_mets_exact_EE.csv')