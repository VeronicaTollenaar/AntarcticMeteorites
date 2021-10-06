# import packages
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.neighbors import KernelDensity

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

# overview
# - open labelled data
# - merge with field sites
# - define plotting function
# - prepare plots
#%%
# open labelled data
f1_radar_mets = pd.read_csv('../Data_Features/radarbackscatter_at_mets.csv')
f2_speed_mets = pd.read_csv('../Data_Features/velocities_at_mets.csv')
f4_slope_mets = pd.read_csv('../Data_Features/slope2km_at_mets.csv')
f5_stemp_mets = pd.read_csv('../Data_Features/stempPERC99_at_mets.csv')

# merge data
data_mets = f1_radar_mets.merge(
            f2_speed_mets).merge(
            f4_slope_mets).merge(
            f5_stemp_mets)
                
# delete individual features
del(f1_radar_mets,
    f2_speed_mets,
    f4_slope_mets,
    f5_stemp_mets)

#%%
# read in abbreviations of meteorite recovery locations
locs_mets = pd.read_csv(
        '../Data_Locations/locations_mets_abbrevs.csv')[[
        'x','y','abbrevs','counts']]
data_mets_locs = data_mets.merge(locs_mets)
#names of 9 largest fieldsites
FSs = ['QUE','MIL','LEW','EET','GRV','ALH','MAC','PCA','FRO','rest']
#%%
# define plotting function
def plotpersite(coln,
                data_mets_locs,
                FSs,
                logx,
                perc,
                bw,
                title,
                xlab,
                savename):
    # arange data into list
    data_mets_locs_list = [data_mets_locs[
        data_mets_locs.abbrevs==FSs[fieldsite]].
        iloc[:,coln].values for fieldsite in range(len(FSs)-1)]
    data_mets_locs_list.append(data_mets_locs[(data_mets_locs.abbrevs!='QUE') &
        (data_mets_locs.abbrevs!='MIL') &
        (data_mets_locs.abbrevs!='LEW') &
        (data_mets_locs.abbrevs!='EET') &
        (data_mets_locs.abbrevs!='GRV') &
        (data_mets_locs.abbrevs!='ALH') &
        (data_mets_locs.abbrevs!='MAC') &
        (data_mets_locs.abbrevs!='PCA') &
        (data_mets_locs.abbrevs!='FRO')].iloc[:,coln].values)
    
    # define values for the x-axis
    x = np.linspace(np.percentile(data_mets_locs.iloc[:,coln],perc)-3*bw,
                    np.percentile(data_mets_locs.iloc[:,coln],100-perc)+3*bw,800)
        
    # define colors and linestyle
    colors = ['#4477aa','#4477aa','#4477aa','#66ccee','#228833','#66ccee','#4477aa','#ccbb44','#ee6677','#aa3377'] #https://personal.sron.nl/~pault/
    linestyles = ['solid',(0, (2, 1)),(0, (5, 1)),'solid','solid',(0, (2, 1)),(0, (3, 1, 1, 1)),'solid','solid','solid']

    # (prepare data for) plot
    fig, ax1 = plt.subplots(figsize=(8,4))
    for fieldn in range(len(data_mets_locs_list)):
        # arange data for kde
        data_for_kde = data_mets_locs_list[fieldn].reshape(-1, 1)
        print(fieldn,FSs[fieldn],
              np.round(np.mean(data_for_kde),2),
              np.round(np.std(data_for_kde),2))
        # perform kde
        kde1 = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data_for_kde)
        # plot kde
        plt.plot(x,np.exp(kde1.score_samples(x.reshape(-1, 1))),
                 label=FSs[fieldn],
                 color=colors[fieldn],
                 linestyle=linestyles[fieldn])
    # plot settings
    if logx==True:
        plt.xscale('log')
    plt.legend()
    plt.subplots_adjust(left=0.2,right=0.8)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('estimated density')
    plt.savefig('../Figures/Features_PerSite_'+savename+'.png',dpi=200)
    plt.show()

#%%
# prepare plots
# radar
plotpersite(coln=2,
            data_mets_locs=data_mets_locs,
            FSs=FSs,
            logx=False,
            perc=0,
            bw=3,
            title='Radar backscatter per field site',
            xlab='radar backscatter (-)',
            savename='radar')
print('radar',np.round(np.percentile(data_mets_locs.iloc[:,2],99),2))

# speed
plotpersite(coln=3,
            data_mets_locs=data_mets_locs,
            FSs=FSs,
            logx=False,
            perc=0,
            bw=0.1,
            title='Surface velocity per field site',
            xlab='surface velocity (m/year)',
            savename='velocities')
print('speed',np.round(np.percentile(data_mets_locs.iloc[:,3],99),2))

# slope
plotpersite(coln=4,
            data_mets_locs=data_mets_locs,
            FSs=FSs,
            logx=False,
            perc=0.1,
            bw=0.002,
            title='Surface slope per field site',
            xlab='surface slope (m/m)',
            savename='slope2km')
print('slope',np.round(np.percentile(data_mets_locs.iloc[:,4],99)*1000,2))

# stemp PERC 99
plotpersite(coln=5,
            data_mets_locs=data_mets_locs,
            FSs=FSs,
            logx=False,
            perc=0,
            bw=0.3,
            title='Surface temperature (99th percentile) per field site',
            xlab='surface temperature ($^\circ$C)',
            savename='stempPERC99')
print('stemp',np.round(np.percentile(data_mets_locs.iloc[:,5],99),2))

#%%
print(np.round(np.mean(data_mets.stemp),2),
      np.round(np.std(data_mets.stemp),2))

print(np.round(np.mean(data_mets.stemp),2),
      np.round(np.std(data_mets.stemp),2))