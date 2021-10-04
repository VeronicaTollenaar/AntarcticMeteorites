# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd
import rasterio

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%
# convert tif file to DataArray

# open tif file
with rasterio.open('../Data_raw/amm125m_v2_200m.tif') as opentif:
    ramp = opentif.read()
    rampmeta = opentif.meta
    
# set missing values to np.nan
ramp = ramp.astype('float')
ramp[ramp==0.] = np.nan

# assign coordinates
xul = rampmeta['transform'][2]+0.5*rampmeta["transform"][0]
yul = rampmeta['transform'][5]-0.5*rampmeta["transform"][0]
res = rampmeta['transform'][0]
(sizey, sizex) = rampmeta['height'], rampmeta['width']
x = np.arange(xul,xul+sizex*res, res)
y = np.arange(yul+sizey*-res+res,yul+res, res)[::-1]

# assign radar as DataArray
radar_DA = xr.DataArray(ramp[0,:,:],dims=['y','x'],
                  coords={'x': x, 'y': y})
# create dataset from DataArray
radar_raw = radar_DA.to_dataset(name='radar')

# delete variables from memory
del(opentif,ramp,rampmeta,radar_DA,res,sizex,sizey,x,xul,y,yul)
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
# extract data at locations where meteorites have been found (gridded)
# extract data at locations that are to be classified
radar_at_mets, radar_at_toclass = extractdata(
    radar_raw,
    'radarbackscatter',
    'radar',
    'locations_mets',
    'locations_toclass')

#%% # extract data at exact finding locations
radar_at_mets, radar_at_toclass = extractdata(
    radar_raw,
    'radarbackscatter_exactlocs',
    'radar',
    'locations_mets_nogrid',
    'locations_toclass')
