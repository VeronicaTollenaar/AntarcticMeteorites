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
# open data
velocities_raw = xr.open_dataset('../Data_raw/antarctic_ice_vel_phase_map_v01.nc')

#%%
# calculate surface velocity at all locations in dataset
velocities_raw['speed'] = np.sqrt(
    (velocities_raw.VX)**2 + (velocities_raw.VY)**2)

#%%
# drop variables before extracting data
velocities_raw_drop = velocities_raw.drop_vars([
    'VX','VY','STDX','STDY','ERRX','ERRY','CNT','SOURCE'])

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
# extract data at locations where meteorites have been found (gridded)
# extract data at locations that are to be classified
speed_at_mets, speed_at_toclass = extractdata(
    velocities_raw_drop,
    'velocities',
    'speed',
    'locations_mets',
    'locations_toclass')

#%% # extract data at exact finding locations
slope_at_mets, slope_at_toclass = extractdata(
    velocities_raw_drop,
    'velocities_exactlocs',
    'speed',
    'locations_mets_nogrid',
    'locations_toclass')

#%%
# function to plot histogram
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

def plothist(
        data_at_all,
        data_at_mets,
        data_at_toclass,
        xlabel,
        title,
        savename,
        bins,
        percentile,
        logx=False):
    
    # define bins
    if logx == True:
        bins = np.logspace(np.log10(np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),percentile)),
               np.log10(np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),100-percentile)),bins)
    else:
        bins = np.linspace(np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),percentile),
               np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),100-percentile),bins)

    # plot figure
    fig, ax1 = plt.subplots(figsize=(8.8/2.54, 7/2.54))
    
    # plot all data
    plot_at_all = ax1.hist(data_at_all,
                           label='entire continent',
                           edgecolor='black',
                           linewidth=0.6,
                           bins=bins,
                           color='black',
                           alpha=0.9,
                           weights=(1/len(data_at_all))*np.ones_like(data_at_all))
    
    # plot data_at_toclass
    plot_at_toclass = ax1.hist(data_at_toclass,
                           label='expanded blue ice areas', 
                           edgecolor='black', 
                           linewidth=0.6,
                           bins=bins, 
                           color='grey',
                           alpha=0.7,
                           weights=(1/len(data_at_toclass))*np.ones_like(data_at_toclass))

    # plot data_at_mets
    plot_at_mets = ax1.hist(data_at_mets,
                           label='meteorite recovery locations',
                           edgecolor='black', 
                           linewidth=0.6,
                           bins=bins, 
                           color='yellow',
                           alpha=0.4,
                           weights=(1/len(data_at_mets))*np.ones_like(data_at_mets))

    # plot xlabel, title and legend
    ax1.set_ylabel('normalized counts')
    ax1.set_xlabel(xlabel)
    ax1.set_title(title)
    ax1.legend()
    if logx == True:
        ax1.set_xscale('log')
    bottom, top = plt.ylim()  # return the current ylim
    plt.ylim((bottom, top*1.45))   # set the ylim to bottom, top
    plt.subplots_adjust(top = 0.92, bottom = 0.15, right = 0.98, left = 0.16, 
            hspace = 0, wspace = 0)

    # save figure
    #plt.tight_layout()
    #plt.subplots_adjust(hspace=0.4)
    plt.savefig('../Figures/'+savename,dpi=300)
    plt.show()
    
#%%
# plot histogram of all values, positive, and unlabelled observations
plothist(
        velocities_raw_drop.speed.values[~np.isnan(velocities_raw_drop.speed.values)],
        speed_at_mets.speed.values,
        speed_at_toclass.speed.values,
        'surface velocity (m/year)',
        'Histogram of surface velocities',
        'Hist_velocities.png',
        bins=30,
        percentile=0.1,
        logx=True)

