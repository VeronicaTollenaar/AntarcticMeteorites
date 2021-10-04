# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# open data
thickness_raw = xr.open_dataset('../Data_raw/BedMachineAntarctica_2020-07-15_v02.nc')

#%%
# drop variables before extracting data
thickness_raw_drop = thickness_raw.drop_vars([
    'mapping','mask','firn','surface','bed','errbed','source','geoid'])

#%%
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
# extract data at locations where meteorites have been found (gridded)
# extract data at locations that are to be classified
thickn_at_mets, thickn_at_toclass = extractdata(
    thickness_raw_drop,
    'icethickness',
    'thickness',
    'locations_mets',
    'locations_toclass')
