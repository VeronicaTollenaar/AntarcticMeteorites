# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

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
aucs_all = pd.DataFrame()
neg = 'SpecNeg' # or 'RandNeg'
for i in range(len(combinations_slope_stemp)):
    aucs_slopestempcomb = read_aucs(neg,combinations_slope_stemp[i][0]+combinations_slope_stemp[i][1])
    aucs_all = aucs_all.append(aucs_slopestempcomb)

# plot ROCs
ax = plt.subplot(111)
# set threshold for the area under the curve
th = 0.88 # for 'RandNeg' try 0.95
# plot all ROCs with AUC>threshold
for k in range(len(aucs_all[aucs_all.average>th])):
    # for legend: number of principal components and number of features
    n_pcs_n_fs = aucs_all[aucs_all.average>th].index[k][0]
    # for legend: combination of features
    fs_combination = aucs_all[aucs_all.average>th].index[k][1]
    # for legend: combination of definition slope and temperature
    version_slopestemp = aucs_all[aucs_all.average>th].SlopeStemp[k]
    # open true positive and false positive data
    try:
        roc_toplot = pd.read_csv('../Results/'+version_slopestemp+neg+'/ROC_average_'+n_pcs_n_fs+fs_combination+'.csv')
    except FileNotFoundError:
        roc_toplot = pd.read_csv('../Results/'+version_slopestemp+neg+'_AllCombinations/ROC_average_'+n_pcs_n_fs+fs_combination+'.csv')
    # plot curve
    plt.plot(roc_toplot.FalsePositive_average,roc_toplot.TruePositive_average,
              label = n_pcs_n_fs + ', ' + fs_combination +
              ', ' + str(np.round(aucs_all[aucs_all.average>th].average[k],3))+
              ', '+version_slopestemp, linewidth = 0.6)
# plot settings
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.title('ROCs with the largest AUC')
plt.xlabel('false positive rate ')
plt.ylabel('true positive rate')
ax.legend(loc='center left', bbox_to_anchor=(0.6, 0.5),fontsize=6)
#plt.savefig('../Figures/ROCs_largest_AUCs_ExhaustiveFeatureSelection.png',dpi=200)
plt.show()
