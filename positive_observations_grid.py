# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt

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
met_locs_reproj_ee = met_locs_reproj_gr.copy()
met_locs_reproj_ee['indexno'] = np.linspace(0,
                                            len(met_locs_reproj_ee)-1,
                                            len(met_locs_reproj_ee))
met_locs_reproj_ee.to_csv('../Data_Locations/locations_mets_EE.csv')

# original observations
met_locs_ee = met_locs[['new_x','new_y']].copy()
met_locs_ee['indexno'] = np.linspace(0,
                                     len(met_locs_ee)-1,
                                     len(met_locs_ee))
met_locs_ee.to_csv('../Data_Locations/locations_mets_exact_EE.csv')

#%%
# select observations by mass

# prepare data
# predefine empty array to save mass
mass = np.zeros(len(met_locs))
# loop through data and store values for mass in grams
for i in range(len(met_locs)):
    if met_locs.Mass_x[i][-3:] == ' kg':
        mass[i] = float(met_locs.Mass_x[i][:-3])*1000
    if met_locs.Mass_x[i][-3:] == ' mg':
        mass[i] = float(met_locs.Mass_x[i][:-3])/1000
    if met_locs.Mass_x[i][-2:] == ' g':
        mass[i] = float(met_locs.Mass_x[i][:-2])
# assign column as mass in grams
met_locs['mass_grams'] = mass

# plot distribution of mass
# define bins
logbins = np.logspace(np.log10(0.1),
                      np.log10(10000),
                      20)
# define figure
ax = plt.subplot(111)
# plot histogram
plt.hist(met_locs.mass_grams,bins=logbins,color='grey',
         edgecolor='black',linewidth=0.2)
# set scale x-axis to logarithmic
plt.xscale('log')
# define title and labels
plt.title('distribution of mass of recovered meteorites')
plt.xlabel('mass (gram)')
plt.ylabel('counts')
# plot vertical lines for three thresholds
plt.vlines(100,0,1800,linestyle='--')
plt.vlines(150,0,1800,linestyle='--')
plt.vlines(200,0,1800,linestyle='--')
# set ylimits
plt.ylim([0,1800])
# save figure
#plt.savefig('../Figures/mass_meteorites.png',dpi=200)
plt.show()

# reproject and save observations heavier than threshold (3x)
# define threshold 
th = [100,150,200] # grams
# loop through thresholds
for i in range(len(th)):
    met_locs_mass = met_locs[met_locs.mass_grams>th[i]]
    # perform reprojection by finding nearest neighbours
    met_locs_mass_reproj = coords_vel.sel(x=met_locs_mass.new_x.to_xarray(),
                              y=met_locs_mass.new_y.to_xarray(),
                              method='nearest')
    # reorganize as dataframe
    met_locs_mass_reproj_df = met_locs_mass_reproj.to_dataframe()[['x','y']]
    met_locs_mass_reproj_df['abbrevs']=met_locs_mass.abbrevs.values
    # groupby to remove duplicates
    met_locs_mass_reproj_gr = met_locs_mass_reproj_df.groupby(
                         ['x','y','abbrevs']).size().reset_index(name='counts')
    # print observations per threshold
    print(th[i],len(met_locs_mass_reproj_gr))
    met_locs_mass_reproj_gr.to_csv('../Data_Locations/locations_mets_heavierthan'+str(th[i])+'grams.csv',index=False)

