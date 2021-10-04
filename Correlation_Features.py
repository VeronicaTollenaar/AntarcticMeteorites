# import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter 
from matplotlib import colors
import xarray as xr
import os
import pandas as pd
from sklearn import preprocessing
import seaborn as sns

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# open labelled data
f1_radar_mets = pd.read_csv('../Data_Features/radarbackscatter_at_mets.csv')
f2_speed_mets = pd.read_csv('../Data_Features/velocities_at_mets.csv')
f3_iceth_mets = pd.read_csv('../Data_Features/icethickness_at_mets.csv')
f4_slope_mets = pd.read_csv('../Data_Features/slope2km_at_mets.csv')
f5_stemp_mets = pd.read_csv('../Data_Features/stempPERC99_at_mets.csv')
f6_disto_mets = pd.read_csv('../Data_Features/distanceoutcrops_at_mets.csv')

# merge data
data_mets = f1_radar_mets.merge(
            f2_speed_mets).merge(
            f3_iceth_mets).merge(
            f4_slope_mets).merge(
            f5_stemp_mets).merge(
            f6_disto_mets)
                
# delete individual features
del(f1_radar_mets,
    f2_speed_mets,
    f3_iceth_mets,
    f4_slope_mets,
    f5_stemp_mets,
    f6_disto_mets)

#%%
# open unlabelled data
f1_radar_toclass = pd.read_csv('../Data_Features/radarbackscatter_at_toclass.csv')
f2_speed_toclass = pd.read_csv('../Data_Features/velocities_at_toclass.csv')
f3_iceth_toclass = pd.read_csv('../Data_Features/icethickness_at_toclass.csv')
f4_slope_toclass = pd.read_csv('../Data_Features/slope2km_at_toclass.csv')
f5_stemp_toclass = pd.read_csv('../Data_Features/stempPERC99_at_toclass.csv')
f6_disto_toclass = pd.read_csv('../Data_Features/distanceoutcrops_at_toclass.csv')

# merge data
data_toclass = f1_radar_toclass.merge(
               f2_speed_toclass).merge(
               f3_iceth_toclass).merge(
               f4_slope_toclass).merge(
               f5_stemp_toclass).merge(
               f6_disto_toclass)

# delete individual features
del(f1_radar_toclass,
    f2_speed_toclass,
    f3_iceth_toclass,
    f4_slope_toclass,
    f5_stemp_toclass,
    f6_disto_toclass)
                   
#%%
# transform features

data_mets_transf = data_mets.copy()
data_toclass_transf = data_toclass.copy()

data_mets_transf['radar']    = data_mets.radar + np.random.RandomState(
                               3).normal(0,0.25,len(data_mets.radar))
data_toclass_transf['radar'] = data_toclass.radar + np.random.RandomState(
                               6).normal(0,0.25,len(data_toclass.radar))

data_mets_transf['speed']    = np.log10(data_mets.speed.values)
data_toclass_transf['speed'] = np.log10(data_toclass.speed.values)

data_mets_transf['slope_max']    = np.log10(data_mets.slope_max.values)
data_toclass_transf['slope_max'] = np.log10(data_toclass.slope_max.values)

data_mets_transf['stemp']    = data_mets.stemp + np.random.RandomState(
                               5).normal(0,0.04,len(data_mets.stemp))
data_toclass_transf['stemp'] = data_toclass.stemp + np.random.RandomState(
                               8).normal(0,0.04,len(data_toclass.stemp))
data_mets_transf['distanceoutcrops']    = np.log10(abs(
                                          data_mets.distanceoutcrops.values +
                                          np.random.RandomState(4).normal(
                                          0,0.1,len(data_mets.distanceoutcrops)
                                          ))+0.1)
data_toclass_transf['distanceoutcrops'] = np.log10(abs(
                                          data_toclass.distanceoutcrops.values +
                                          np.random.RandomState(7).normal(
                                          0,0.1,len(data_toclass.distanceoutcrops)
                                          ))+0.1)
#%%
# reorganize the data into groups of:
# - labelled training data
# - unlabelled training data
# - all data

# define positve labelled training data
train_lab = data_mets_transf[:]


# define unlabelled training data
# exclude positive labeled data from unlabeled data
lab_to_exclude = data_mets_transf.copy()[['x','y']]
lab_to_exclude['label'] = 1
lab_to_exclude_merged = data_toclass_transf.merge(lab_to_exclude,
                                                  how='outer',
                                                  on =['x','y'])
train_unlab = lab_to_exclude_merged[
              np.isnan(lab_to_exclude_merged.label)].drop(
              ['label'],
              axis=1)[:]

# combine all training data into one array
train_all = pd.concat([train_lab,train_unlab])

del(data_mets_transf,
    lab_to_exclude,
    lab_to_exclude_merged,
    data_toclass_transf)
#%%
# standardize features
# use index [:,2:] as the first two columns are the x and y coordinates
scaler_train = preprocessing.StandardScaler().fit(train_all.iloc[:,2:].values)
train_lab_st = scaler_train.transform(train_lab.iloc[:,2:].values)
train_unlab_st = scaler_train.transform(train_unlab.iloc[:,2:].values)
train_all_st = scaler_train.transform(train_all.iloc[:,2:].values)

del(train_lab,train_unlab,scaler_train)

#%%
def correlation_heatmap(train,title):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True,
                cbar_kws={"shrink": .70}, cmap='PRGn',
                vmin = -1, vmax = 1)
    plt.yticks(rotation=0) 
    plt.title(title)
    fig.savefig('../Figures/correlations_'+title+'.png',dpi=200)
    plt.show();

correlation_heatmap(pd.DataFrame(train_all_st,columns=
            ['radar','speed','ice thickness','slope','temperature','dist outcrops']),
            '6features')
