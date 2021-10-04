# import packages
import os
import pandas as pd

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# script reorganizes data extracted from google Earth Engine (script 
#    "ExtractSurfaceTemperatureModis" https://code.earthengine.google.com/bbb3e4705236597c83b7deb291b32512)
# define percentile of Surface Temperature data
perc = '99'
# import csv (extracted from google earth engine)
stemp_at_toclass_unsorted = pd.read_csv(
    '../Data_raw/SurfaceTemperature/ST_perc'+perc+'_toclass.csv')
stemp_at_mets_unsorted = pd.read_csv(
    '../Data_raw/SurfaceTemperature/ST_perc'+perc+'_mets.csv')
stemp_at_mets_exact_unsorted = pd.read_csv(
    '../Data_raw/SurfaceTemperature/ST_perc'+perc+'_mets_exact.csv')

# import locations to sort data
locs_toclass = pd.read_csv(
    '../Data_Locations/locations_toclass_EE.csv')
locs_mets = pd.read_csv(
    '../Data_Locations/locations_mets_EE.csv')
locs_mets_exact = pd.read_csv(
    '../Data_Locations/locations_mets_exact_EE.csv')

# merge temperature data with location data
stemp_at_toclass = pd.merge(locs_toclass, stemp_at_toclass_unsorted, on='indexno')
stemp_at_mets = pd.merge(locs_mets, stemp_at_mets_unsorted, on='indexno')
stemp_at_mets_exact = pd.merge(locs_mets_exact, stemp_at_mets_exact_unsorted, on='indexno')

# drop unnecessary columns
stemp_at_toclass = stemp_at_toclass[['x','y','last']]
stemp_at_mets = stemp_at_mets[['x','y','last']]
stemp_at_mets_exact = stemp_at_mets_exact[['x','y','last']]


# rename columns
stemp_at_toclass = stemp_at_toclass.rename(columns={"last": "stemp"})
stemp_at_mets = stemp_at_mets.rename(columns={"last": "stemp"})
stemp_at_mets_exact = stemp_at_mets_exact.rename(columns={"last": "stemp"})

# drop nans
stemp_at_toclass = stemp_at_toclass.dropna()

# change unit from 0.02 K to Celcius
stemp_at_toclass.stemp = (stemp_at_toclass.stemp*0.02)-273.15
stemp_at_mets.stemp = (stemp_at_mets.stemp*0.02)-273.15
stemp_at_mets_exact.stemp = (stemp_at_mets_exact.stemp*0.02)-273.15


# export data
name = 'stempPERC'+perc
stemp_at_toclass.to_csv('../Data_Features/'+name+'_at_toclass.csv',
                header=True,index=False,columns=['x','y','stemp'])
stemp_at_mets.to_csv('../Data_Features/'+name+'_at_mets.csv',
                header=True,index=False,columns=['x','y','stemp'])
stemp_at_mets_exact.to_csv('../Data_Features/'+name+'_exactlocs_at_mets.csv',
                header=True,index=False,columns=['x','y','stemp'])



