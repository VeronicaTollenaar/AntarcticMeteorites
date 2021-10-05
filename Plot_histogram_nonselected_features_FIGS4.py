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
f3_iceth_mets = pd.read_csv('../Data_Features/icethickness_at_mets.csv')
f6_disto_mets = pd.read_csv('../Data_Features/distanceoutcrops_at_mets.csv')

#%%
# open unlabelled data
f3_iceth_toclass = pd.read_csv('../Data_Features/icethickness_at_toclass.csv')
f6_disto_toclass = pd.read_csv('../Data_Features/distanceoutcrops_at_toclass.csv')

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
font = {'family' : 'Arial', # normally Calibri
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
# plot histograms
fig, axs = plt.subplots(1,2,figsize=(8.8/2.54, 4.5/2.54))

# icethickness
plothist(
        f3_iceth_mets.thickness.values,
        f3_iceth_toclass.thickness.values,
        'ice thickness (m)',
        nbins=20,
        percentile=0.1,
        axis=axs[0],
        logx=False)
# set number of xticks
plt.locator_params(nbins=10)
# adjust xticks
axs[0].xaxis.set_major_locator(plt.MaxNLocator(4))
# plot ylabel
axs[0].set_ylabel('normalized counts')
# plot legend
axs[0].legend(['blue ice','meteorites'],
                labelspacing=0.15,
                handlelength=1,
                handletextpad=0.4,
                borderaxespad=0.15,
                borderpad=0.25)
# plot panel label A
axs[0].text(0.22, 0.95, 'A', transform=axs[0].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# dist to outcrops
plothist(
        f6_disto_mets.distanceoutcrops.values+0.1,
        f6_disto_toclass.distanceoutcrops.values+0.1,
        'dist. to outcrops + 0.1 (km)',
        nbins=20,
        percentile=0.1,
        axis=axs[1],
        logx=True)
# set number of xticks
axs[1].locator_params(axis='x', numticks=5)
# adjust yticks
axs[1].set_yticks((0,0.1,0.2))
# plot panel label B
axs[1].text(0.15, 0.95, 'B', transform=axs[1].transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# adjust location of labels axes
axs[0].xaxis.set_label_coords(0.5,-0.22)
axs[1].xaxis.set_label_coords(0.5,-0.22)

# define title figure
fig.suptitle('Features not considered for final classification',x=0.5,y=0.98)
# adjust whitespace
plt.subplots_adjust(top = 0.88, bottom = 0.22, right = 0.96, left = 0.14, 
            hspace = 0.5, wspace = 0.25)
# save figure
plt.savefig('../Figures/histograms_nonselected2',dpi=300)

