# import packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# open labelled data
f1_radar_mets = pd.read_csv('../Data_Features/radarbackscatter_at_mets.csv')
f2_speed_mets = pd.read_csv('../Data_Features/velocities_at_mets.csv')
f4_slope_mets = pd.read_csv('../Data_Features/slope2km_at_mets.csv')
f5_stemp_mets = pd.read_csv('../Data_Features/stempPERC99_at_mets.csv')

#%%
# open unlabelled data
f1_radar_toclass = pd.read_csv('../Data_Features/radarbackscatter_at_toclass.csv')
f2_speed_toclass = pd.read_csv('../Data_Features/velocities_at_toclass.csv')
f4_slope_toclass = pd.read_csv('../Data_Features/slope2km_at_toclass.csv')
f5_stemp_toclass = pd.read_csv('../Data_Features/stempPERC99_at_toclass.csv')

#%%
# function to plot histogram
def plothist(
        data_at_mets,
        data_at_toclass,
        xlabel,
        nbins,
        percentile, # to avoid outliers cropping the xaxis
        axis,
        logx=False,
        ):
    
    # define bins
    if logx == True:
        bins = np.logspace(np.log10(np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),percentile)),
               np.log10(np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),100-percentile)),nbins)
    else:
        bins = np.linspace(np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),percentile),
               np.percentile(np.concatenate((
                   data_at_toclass,data_at_mets)),100-percentile),nbins)
       
    # plot data_at_toclass
    plot_at_toclass = axis.hist(data_at_toclass,
                           label='expanded blue ice areas', 
                           edgecolor='black', 
                           linewidth=0.6,
                           bins=bins, 
                           color='grey',
                           alpha=0.7,
                           weights=(1/len(data_at_toclass))*np.ones_like(data_at_toclass))

    # plot data_at_mets
    plot_at_mets = axis.hist(data_at_mets,
                           label='meteorite recovery locations',
                           edgecolor='black', 
                           linewidth=0.6,
                           bins=bins, 
                           color='yellow',
                           alpha=0.4,
                           weights=(1/len(data_at_mets))*np.ones_like(data_at_mets))

    # set ylabel and xlabel
    axis.set_ylabel('')
    axis.set_xlabel(xlabel)
    
    # set x-axis to logarithmic scale
    if logx == True:
        axis.set_xscale('log')
    
#%%
# set font for plot
font = {'family': 'Arial', # normally Calibri
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

# plot histograms
fig, axs = plt.subplots(2,2,figsize=(8.8/2.54, 10/2.54))

# stemp
plothist(
        f5_stemp_mets.stemp.values,
        f5_stemp_toclass.stemp.values,
        'surface temperature ($^o$C)',
        nbins=20,
        percentile=0.1,
        axis=axs[0,0],
        logx=False)
plt.locator_params(nbins=10)
# plot ylabel
axs[0,0].set_ylabel('normalized counts')
# plot panel label A
axs[0,0].text(0.15, 0.95, 'A', transform=axs[0,0].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# speed
plothist(
        f2_speed_mets.speed.values,
        f2_speed_toclass.speed.values,
        'surface velocity (m/year)',
        nbins=20,
        percentile=0.1,
        axis=axs[0,1],
        logx=True)
# adjust number of xticks
axs[0,1].locator_params(axis='x', numticks=10)
# plot legend
axs[0,1].legend(['blue ice','meteorites'],
                labelspacing=0.15,
                handlelength=1,
                handletextpad=0.4,
                borderaxespad=0.15,
                borderpad=0.25)
# plot panel label B
axs[0,1].text(0.15, 0.95, 'B', transform=axs[0,1].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# radar
plothist(
        f1_radar_mets.radar.values,
        f1_radar_toclass.radar.values,
        'radar backscatter (-)',
        nbins=20,
        percentile=0.01,
        axis=axs[1,0],
        logx=False)
# plot panel label C
axs[1,0].text(0.13, 0.95, 'C', transform=axs[1,0].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# slope
plothist(
        f4_slope_mets.slope_max.values,
        f4_slope_toclass.slope_max.values,
        'surface slope (m/m)',
        nbins=20,
        percentile=0.1,
        axis=axs[1,1],
        logx=True)
# adjust number of xticks
axs[1,1].locator_params(axis='x', numticks=5)
# plot panel label D
axs[1,1].text(0.16, 0.95, 'D', transform=axs[1,1].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# adjust xticks (for non-log)
axs[0,0].xaxis.set_major_locator(plt.MaxNLocator(5))
axs[1,0].xaxis.set_major_locator(plt.MaxNLocator(5))

# adjust yticks for all subplots
axs[0,1].set_yticks((0,0.1,0.2,0.3))
axs[1,0].set_yticks((0,0.1))
axs[1,1].set_yticks((0,0.1,0.2))

# adjust location of labels axes
axs[0,0].xaxis.set_label_coords(0.5,-0.2)
axs[0,1].xaxis.set_label_coords(0.5,-0.22)
axs[1,0].xaxis.set_label_coords(0.5,-0.2)
axs[1,1].xaxis.set_label_coords(0.5,-0.2)

axs[0,0].yaxis.set_label_coords(-0.21,0.5)

# define title figure
fig.suptitle('Selected features',x=0.5,y=0.98)
# adjust whitespace
plt.subplots_adjust(top = 0.93, bottom = 0.1, right = 0.99, left = 0.12, 
            hspace = 0.5, wspace = 0.25)
# save figure
plt.savefig('../Figures/histograms.png',dpi=300)
