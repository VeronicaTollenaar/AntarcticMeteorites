# import packages
import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# function tp read in all AUCs per combination of slope and temperature definition
def read_aucs(neg,foldername_part1):
    # compose list of files that contain AUCs in folder of analyses with temperature and velocity (TemperatureVelocity)
    listfiles_TV = os.listdir('../Results/'+foldername_part1+neg)
    # compose list of files that contain AUCs in folder of analyses all other combinations (AllCombinations)
    try:
        listfiles_AC = os.listdir('../Results/'+foldername_part1+neg+'_AllCombinations')
    except FileNotFoundError:
        listfiles_AC = []
    
    # loop through all files that contain "AUCs" in both directories
    # select files that contain "AUCs" in listfiles_TV
    AUCinname_TV = [s for s in listfiles_TV if "AUCs" in s]
    
    # select files that contain "AUCs" in listfiles_AC
    AUCinname_AC = [s for s in listfiles_AC if "AUCs" in s]
    
    # all filenames
    AUCinname = set(AUCinname_AC + AUCinname_TV)
    
    # define empty dataframe to save values of the area under the curve in a loop
    aucs = pd.DataFrame()
    # open all results (not every file in both folders, therefore try function is used)
    for filename in AUCinname:
        try: 
            res_part1 = pd.read_csv('../Results/'+foldername_part1+neg+'/'+filename)
        except FileNotFoundError:
            res_part1 = pd.DataFrame()
        try:
            res_part2 = pd.read_csv('../Results/'+foldername_part1+neg+'_AllCombinations/'+filename)
            res_ = pd.concat([res_part1,res_part2],axis=1)
        except FileNotFoundError:
            res_ = res_part1
        # append newly opened results to dataframe
        aucs_addedline = pd.concat({filename[5:-4]:res_.drop(['Unnamed: 0'],axis=1)},
                          axis=1)
        aucs = pd.concat([aucs,aucs_addedline],axis=1)
    # set index of dataframe
    aucs = aucs.set_index(res_.iloc[:,0])
    # calculate standard deviation of the area under the curve
    aucs.loc['std_aucs',:] = np.std(aucs.iloc[0:10,:],axis=0)
    # transpose dataframe
    aucs_transposed = aucs.T
    # add a column that indicates the version of the slope and temperature definition
    aucs_transposed['SlopeStemp'] = foldername_part1
    # reset index (not needed in this script)
    aucs_all = aucs_transposed
    del(listfiles_TV, listfiles_AC)
    return(aucs_all)

# define different combinations of slope and stemp
combinations_slope_stemp = [['slope5km','stempPERC70'],
                            ['slope2km','stempPERC70'],
                            ['slope400m','stempPERC70'],
                            ['slope5km','stempPERC90'],
                            ['slope2km','stempPERC90'],
                            ['slope400m','stempPERC90'],
                            ['slope5km','stempPERC95'],
                            ['slope2km','stempPERC95'],
                            ['slope400m','stempPERC95'],
                            ['slope5km','stempPERC99'],
                            ['slope2km','stempPERC99'],
                            ['slope400m','stempPERC99']]

# read in all AUCs per combination
aucs_specneg = pd.DataFrame()
aucs_randneg = pd.DataFrame()
for i in range(len(combinations_slope_stemp)):
    aucs_slopestempcomb_specneg = read_aucs('SpecNeg',combinations_slope_stemp[i][0]+combinations_slope_stemp[i][1])
    aucs_specneg = aucs_specneg.append(aucs_slopestempcomb_specneg)
    aucs_slopestempcomb_randneg = read_aucs('RandNeg',combinations_slope_stemp[i][0]+combinations_slope_stemp[i][1])
    aucs_randneg = aucs_randneg.append(aucs_slopestempcomb_randneg)
# reset indices
aucs_specneg = aucs_specneg.reset_index()
aucs_randneg = aucs_randneg.reset_index()
#%%
# select only results that use the a certain combination of features (e.g., 0134) 0=radar, 1=velocity, 2=icethickness, 3=slope, 4=temperature, 5=distance to outcrops
aucs_set_features = aucs_specneg[aucs_specneg.level_1=='0134']
aucs_set_features_randneg = aucs_randneg[aucs_randneg.level_1=='0134']
#%%
# influence percentile temperature
# set font properties
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

# define figure
fig, ax1 = plt.subplots(figsize=(8.8/2.54, 8/2.54))

# define barwidth
bw = 0.2
# define locations for bars
x = np.arange(11)
# define colors
colors = plt.cm.PuRd(np.linspace(0.35,1,4))

# plot bars per percentile
plt.bar(x-1.5*bw,aucs_set_features[aucs_set_features.SlopeStemp=='slope2kmstempPERC70'].values[0,2:-2],
                                    width=bw,color = colors[0],label='70th')
plt.bar(x-0.5*bw,aucs_set_features[aucs_set_features.SlopeStemp=='slope2kmstempPERC90'].values[0,2:-2],
                                    width=bw,color = colors[1],label='90th')
plt.bar(x+0.5*bw,aucs_set_features[aucs_set_features.SlopeStemp=='slope2kmstempPERC95'].values[0,2:-2],
                                    width=bw,color = colors[2],label='95th')
plt.bar(x+1.5*bw,aucs_set_features[aucs_set_features.SlopeStemp=='slope2kmstempPERC99'].values[0,2:-2],
                                    width=bw,color = colors[3],label='99th')
# plot settings
plt.ylim([0.5,1])
# plot legend and labels
plt.legend(ncol=4,
           labelspacing=0,
           columnspacing=1,
           handletextpad=0.1)
plt.xticks(x,
          (aucs_set_features.columns[2:-2]),rotation=90)
plt.title('Influence percentile temperature')
plt.ylabel('Area Under the ROC Curve (AUC)')
# adjust spacing and save figure
fig.subplots_adjust(top = 0.93, bottom = 0.18, right = 0.99, left = 0.14, 
        hspace = 0.8, wspace = 0.8)
#plt.savefig('../Figures/influence_percentile.png',dpi=300)
plt.show()
# print comparison of values
perc_1 = '70' # compare values of perc_1 percentile
perc_2 = '99' # to values of perc_2 percentile
print('reduction of AUC obtained with non-MSZ data of '+perc_1+ ' percentile',
          aucs_set_features[aucs_set_features.SlopeStemp=='slope2kmstempPERC'+perc_1].T.iloc[2:-2]/ \
          aucs_set_features[aucs_set_features.SlopeStemp=='slope2kmstempPERC'+perc_2].T.iloc[2:-2].values, 
          'compared to '+perc_2+'th percentile')    

#%%
# influence slope
# set font properties
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

# define figure
fig, ax1 = plt.subplots(figsize=(8.8/2.54, 8/2.54))
# use same barwidth and spacing as previous figure
# define colors
colors = plt.cm.summer(np.linspace(0,0.65,3))

# plot bars per surface slope distance
plt.bar(x-1*bw,aucs_set_features[aucs_set_features.SlopeStemp=='slope400mstempPERC99'].values[0,2:-2],
        width=bw,color = colors[0],label='0.4 km')
plt.bar(x,aucs_set_features[aucs_set_features.SlopeStemp=='slope2kmstempPERC99'].values[0,2:-2],
        width=bw,color = colors[1],label='2.2 km')
plt.bar(x+1*bw,aucs_set_features[aucs_set_features.SlopeStemp=='slope5kmstempPERC99'].values[0,2:-2],
        width=bw,color = colors[2],label='5 km')

# plot settings
plt.ylim([0.5,1])
# plot legend and labels
plt.title('Influence surface slope length scale')
plt.legend(ncol=3,
           labelspacing=0,
           columnspacing=1,
           handletextpad=0.1)
plt.xticks(x,
          (aucs_set_features.columns[2:-2]),rotation=90)
plt.ylabel('Area Under the ROC Curve (AUC)')
# ajdust spacing and save figure
fig.subplots_adjust(top = 0.93, bottom = 0.18, right = 0.99, left = 0.14, 
        hspace = 0.8, wspace = 0.8)
#plt.savefig('../Figures/influence_slope.png',dpi=300)
plt.show()

# print comparison of values
distslope_1 = '400m' # compare values of distslope_1 distance over which slope is calculated
distslope_2 = '2km' # to values of distslope_2 distance over which slope is calculated
print('reduction of AUC obtained with non-MSZ data of '+distslope_1,
          aucs_set_features[aucs_set_features.SlopeStemp=='slope'+distslope_1+'stempPERC99'].T.iloc[2:-2]/ \
          aucs_set_features[aucs_set_features.SlopeStemp=='slope'+distslope_2+'stempPERC99'].T.iloc[2:-2].values, 
          'compared to '+distslope_2)
#%%
# plot 2d image of all different slope and temperature combinations
# define names of slopes
slopes = ['400m','2km','5km']
# define names of percentiles
percs = ['70','90','95','99']
# define empty array to fill with values
tofill = np.zeros((4,3))
# loop through all different combinations to fill array with AUC values
for k in range(4):
    for m in range(3):
        tofill[k,m] = aucs_set_features[aucs_set_features.SlopeStemp ==
            'slope'+slopes[m]+'stempPERC'+percs[k]].average
# plot the values of the different combinations of definitions
ax = sns.heatmap(tofill,annot=True)
# plot ticks on x- and y-axis
plt.yticks(np.arange(4)+0.5,percs)
plt.xticks(np.arange(3)+0.5,slopes)

