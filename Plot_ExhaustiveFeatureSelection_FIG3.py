# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
## read/prepare data
# define selected definition of slope and surface temperature
combinations_slope_stemp = [['slope5km','stempPERC99']]
# define first part of name of folder with results
foldername_part1 = combinations_slope_stemp[0][0] + combinations_slope_stemp[0][1]

# function tp read in all AUCs per combination
def read_aucs(neg):
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
    # reset index
    aucs_all = aucs_transposed.reset_index()
    return(aucs_all)

# read in results with negative validation data ('SpecNeg') and random validation data ('RandNeg')
aucs_specneg = read_aucs('SpecNeg')
aucs_randneg = read_aucs('RandNeg')
# define list with numbers referring to different features
fs = ['0','1','2','3','4','5']
# define empty array to store average value AUC of all classifications obtained WITH a certain feature (for negative validation data)
average_auc_with_f_specneg = np.zeros(len(fs))
# loop through all features
for k in range(len(fs)):
    # initialize a counter for the number of classifications WITH a feature
    classifications = 0 
    # initialize a counter for the total area under the curve
    total_auc = 0
    # loop through all aucs obtained in the classifications
    for i in range(len(aucs_specneg)):
        # add auc to total auc if a certain feature is used for the classification and count number of classifications
        if fs[k] in aucs_specneg.level_1.iloc[i]:
            total_auc = total_auc + aucs_specneg.average.iloc[i]
            classifications = classifications + 1
    # print number of classifications (as check)
    print(classifications)
    # calculate average auc by dividing the total auc by the number of classifications
    average_auc_with_f_specneg[k] = total_auc/classifications
  
# define empty array to store average value AUC of all classifications obtained WITHOUT a certain feature (for negative validation data)
average_auc_without_f_specneg = np.zeros(len(fs))
# loop through all features
for k in range(len(fs)):
    # initialize a counter for the number of classifications WITHOUT a feature
    classifications = 0 
    # initialize a counter for the total area under the curve
    total_auc = 0
    # loop through all aucs obtained in the classifications
    for i in range(len(aucs_specneg)):
        # add auc to total auc if a certain feature is NOT used for the classification and count number of classifications
        if fs[k] not in aucs_specneg.level_1.iloc[i]:
            total_auc = total_auc + aucs_specneg.average.iloc[i]
            classifications = classifications + 1
    # print number of classifications (as check)
    print(classifications)
    # calculate average auc by dividing the total auc by the number of classifications
    average_auc_without_f_specneg[k] = total_auc/classifications
   
# define empty array to store average value AUC of all classifications obtained WITH a certain feature (for random validation data)
average_auc_with_f_randneg = np.zeros(len(fs))
# loop through all features
for k in range(len(fs)):
    # initialize a counter for the number of classifications WITH a feature
    classifications = 0 
    # initialize a counter for the total area under the curve
    total_auc = 0
    # loop through all aucs obtained in the classifications
    for i in range(len(aucs_randneg)):
        # add auc to total auc if a certain feature is used for the classification and count number of classifications
        if fs[k] in aucs_randneg.level_1.iloc[i]:
            total_auc = total_auc + aucs_randneg.average.iloc[i]
            classifications = classifications + 1
    # print number of classifications (as check)
    print(classifications)
    # calculate average auc by dividing the total auc by the number of classifications
    average_auc_with_f_randneg[k] = total_auc/classifications
        
# define empty array to store average value AUC of all classifications obtained WITHOUT a certain feature (for random validation data)
average_auc_without_f_randneg = np.zeros(len(fs))
# loop through all features
for k in range(len(fs)):
    # initialize a counter for the number of classifications WITH a feature
    classifications = 0 
    # initialize a counter for the total area under the curve
    total_auc = 0
    # loop through all aucs obtained in the classifications
    for i in range(len(aucs_randneg)):
        # add auc to total auc if a certain feature is used for the classification and count number of classifications
        if fs[k] not in aucs_randneg.level_1.iloc[i]:
            total_auc = total_auc + aucs_randneg.average.iloc[i]
            classifications = classifications + 1
    # print number of classifications (as check)
    print(classifications)
    # calculate average auc by dividing the total auc by the number of classifications
    average_auc_without_f_randneg[k] = total_auc/classifications


#%%
## plot figure
# figure size
fig = plt.figure(figsize=(18./2.54, 14/2.54))
gs = fig.add_gridspec(2, 2,height_ratios=[1,1.2])
# plot fontsetting
font = {'family' : 'Arial', # normally Calibri
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

## ------ plot panel A
ax2 = fig.add_subplot(gs[0, :])
ax2.zorder = 2
# make white background transparent
ax2.patch.set_alpha(0)

# prepare data
# rename columns dataframe with aucs obtained with negative validation data
aucs_specneg = aucs_specneg.rename(columns={"QUE":"QUE_SpecNeg",
                             "MIL":"MIL_SpecNeg",
                             "LEW":"LEW_SpecNeg",
                             "EET":"EET_SpecNeg",
                             "GRV":"GRV_SpecNeg",
                             "ALH":"ALH_SpecNeg",
                             "MAC":"MAC_SpecNeg",
                             "PCA":"PCA_SpecNeg",
                             "FRO":"FRO_SpecNeg",
                             "rest":"rest_SpecNeg",
                             "average":"average_SpecNeg",
                             "std_aucs":"std_aucs_SpecNeg"})
# merge dataframes with aucs obtained with random validation data and with negative validation data
aucs_allneg = pd.merge(aucs_randneg,aucs_specneg)

# sort values
aucs_allneg = aucs_allneg.sort_values(by=['average_SpecNeg'])

# define linewidth for errorbars/scattered markers
lw=0.5

# scatter specific negative data AUCs + stds
ax2.scatter(np.linspace(0.25,len(aucs_allneg)/2 - 0.25,len(aucs_allneg)),aucs_allneg.average_SpecNeg,
            color='navy',marker ='d',s=10,
            label='negative data',linewidth=lw)
ax2.errorbar(np.linspace(0.25,len(aucs_allneg)/2 - 0.25,len(aucs_allneg)),aucs_allneg.average_SpecNeg,
              yerr=aucs_allneg.std_aucs_SpecNeg,color='navy',
              linestyle='',linewidth=lw)

# scatter random negative data AUCs + stds
ax2.scatter(np.linspace(0.25,len(aucs_allneg)/2 - 0.25,len(aucs_allneg)),aucs_allneg.average,
            color='cornflowerblue',marker='.',
            label='random data',linewidth=lw)
ax2.errorbar(np.linspace(0.25,len(aucs_allneg)/2 - 0.25,len(aucs_allneg)),aucs_allneg.average,
             yerr=aucs_allneg.std_aucs,
             color='cornflowerblue',linestyle='',linewidth=lw)
# plot labels and arrange positions
ax2.set_ylabel('Area Under the \nROC Curve (AUC)')
ax2.yaxis.set_label_coords(-0.08,0.7)
ax2.yaxis.set_label_position("left")
ax2.yaxis.tick_left()
ax2.set_ylim([-0.3,1])
ax2.set_yticks(np.linspace(0.25,1,4))
ax2.legend(loc='center right',handletextpad=0.1)
# x-axis settings
ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax2.set_xlim(0,len(aucs_allneg)*0.5)

# plot feature combination
ax1 = ax2.twinx()
# define list with numbers referring to different features
fs = ['0','1','2','3','4','5']
# define colors (corresponding to Figure 2)
colors = ['#c9c9c9',
          '#fdd0a2',
          '#e8b3d3',
          '#c7e9c0',
          '#99dcd2',
          '#b9acde']
# change order of features (hardcoded)
difforder = (5,2,1,3,4,0)

# plot feature combinations
for p in range(len(aucs_allneg)):
    for a in range(len(fs)):
        if fs[difforder[a]] in aucs_allneg.level_1.iloc[p]:
            rectangle = plt.Rectangle((p*0.5,float(a)*0.5),0.5,0.5,
                                      color=colors[difforder[a]],
                                      zorder= 0,
                                      linewidth=0)
            plt.gca().add_patch(rectangle)

# plot vertical and horizontal lines
ax1.vlines(np.linspace(0,len(aucs_allneg)/2,len(aucs_allneg)+1),0,8,
           linewidth=0.15,zorder=1,color='k')   
ax1.hlines(3,0,len(aucs_allneg)/2,linewidth=1,color='k')

# plot labels and arrange positions
ax1.set_ylim([0,8])
ax1.yaxis.set_label_position("left")
ax1.yaxis.tick_left()
plt.yticks(np.linspace(0.25,2.75,6),
          ( 'Distance to outcrops',
            'Ice thickness',
            'Surface velocity',
            'Surface slope',
            'Surface temperature',
            'Radar backscatter'))

# plot title
plt.title('Exhaustive feature selection')
# plot panel label A
ax1.text(0.025, 1.09, 'A', transform=ax1.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')


## ------ plot panel B
ax3 = fig.add_subplot(gs[1:, 0])
# calculate relative improvement both for average aucs obtained with negative validation data and random validation data
toplot_specneg = average_auc_with_f_specneg/average_auc_without_f_specneg
toplot_randneg = average_auc_with_f_randneg/average_auc_without_f_randneg
# define labels of features
fs_labels = ['Radar backscatter',
             'Surface velocity',
             'Ice thickness',
             'Surface slope',
             'Surface temperature',
             'Distance to outcrops']
# compose dataframe with relative improvement and corresponding labels
relative_improvement = pd.DataFrame({'specneg': toplot_specneg,
                                     'randneg': toplot_randneg,
                                     'features': fs_labels})
# sort values
relative_improvement = relative_improvement.sort_values(
    by=['specneg'],ascending=True)

# define bar width
bw = 0.2
# define locations to plot bars
x = np.arange(6)

# plot bars of relative improvement for negative and random validation data
plt.barh(x+bw,relative_improvement.specneg,bw*2,
        label='negative data',
        color='navy')
plt.barh(x-bw,relative_improvement.randneg,bw*2,
        label='random data',
        color='cornflowerblue')
# plot a thresholding line at 1.0
plt.vlines(1,-3*bw,5+3*bw,linewidth=0.6,linestyle='-',color='k')

# plot labels and arange positions
plt.xlim([0.82,1.42])
plt.ylim([-3*bw,5+3*bw])
plt.yticks(np.arange(6),
          (relative_improvement.features),rotation=0)
plt.xlabel('with feature/without feature')
plt.legend()
plt.title('Improvement average AUC')
# plot panel label B
ax3.text(0.05, 1.09, 'B', transform=ax3.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

## ------ plot panel C
ax4 = fig.add_subplot(gs[1:, 1:])
ax4.set_aspect('equal', 'box')
# open 3 ROCs
# feature numbers stand for 0 = radar, 1= speed, 2 = ice thickness, 3 = slope, 4 = temp, 5 = dist to outcr
ROC_0 = pd.read_csv('../Results/slope2kmstempPERC99SpecNeg/ROC_average_4pcs4fs0134.csv')
ROC_1 = pd.read_csv('../Results/slope2kmstempPERC99SpecNeg_AllCombinations/ROC_average_4pcs4fs0125.csv')
ROC_2 = pd.read_csv('../Results/slope2kmstempPERC99SpecNeg_AllCombinations/ROC_average_1pcs1fs3.csv')
# add 1,1, to ROC_1 and ROC_0
ROC_0 = ROC_0.append(pd.DataFrame({"TruePositive_average":[1], 
                    "FalsePositive_average":[1]}) )
ROC_1 = ROC_1.append(pd.DataFrame({"TruePositive_average":[1], 
                    "FalsePositive_average":[1]}) )

# plot ROCs
ax4.plot(ROC_2.FalsePositive_average,ROC_2.TruePositive_average,
         color='brown',linestyle='--',label='example 1')
ax4.plot(ROC_1.FalsePositive_average,ROC_1.TruePositive_average,
         color='navy',label='example 2',linestyle='--')
ax4.plot(ROC_0.FalsePositive_average,ROC_0.TruePositive_average,
         color='navy',label='example 3')
# plot filling under ROCs
ax4.fill_between(ROC_0.FalsePositive_average,ROC_0.TruePositive_average,0,
                 color='navy',
                 alpha=0.2)
ROC_fill = ROC_1.append(pd.DataFrame({"TruePositive_average":[0], 
                    "FalsePositive_average":[1]}) )
ax4.fill(ROC_fill.FalsePositive_average,ROC_fill.TruePositive_average,
                 color='navy',
                 fill=False,
                 alpha=0.05,
                 hatch='xxxx')

# plot operating point
OP = (ROC_0.FalsePositive_average[424],ROC_0.TruePositive_average[424]) #(value hard coded, from Classify_observations.py)
ax4.scatter(OP[0],OP[1],color='navy',s=20)

# plot vertical & horizontal lines in panel A for seleted classifications
pt = 0.05 # spacing
ax1.vlines((3,3.5),0,3,
           linewidth=2,zorder=1,
           color='brown',linestyle='--') 
ax1.hlines((0,3),3-pt,3.5+pt,
           linewidth=2,zorder=2,
           color='brown',linestyle='--')

ax1.vlines((12.5,13),0,3,
           linewidth=2,zorder=1,
           color='navy',linestyle='--') 
ax1.hlines((0,3),12.5-pt,13+pt,
           linewidth=2,zorder=2,
           color='navy',linestyle='--') 

ax1.vlines((29.5,30),0,3,
           linewidth=2,zorder=1,
           color='navy',linestyle='-') 
ax1.hlines((0,3),29.5-pt,30+pt,
           linewidth=2,zorder=2,
           color='navy',linestyle='-') 

# adjust number of x and y labels
plt.locator_params(nbins=5)
# annotate lines
ax4.annotate(r'$\approx$random classification',xy=(0.11,0.05),
             rotation=45, size=9,color='brown')
ax4.annotate('better',xy=(0.5,0.79),
             rotation=-45, size=9,color='k')

# draw arrows better/worse
plt.annotate('',
xy=(0.46, 0.9), 
arrowprops=dict(arrowstyle='->'), 
xytext=(0.68, 0.68))
ax4.annotate('worse',xy=(0.75,0.54),
              rotation=-45, size=9,color='k')
plt.annotate('',
xy=(0.9, 0.46), 
arrowprops=dict(arrowstyle='->'), 
xytext=(0.68, 0.68))

# draw AUC>AUC
t = plt.annotate('AUC',xy=(0.44,0.2),size=9, 
             color='navy')
t.set_bbox(dict(facecolor='w', alpha=1, edgecolor='navy'))
t2 = plt.annotate('AUC',xy=(0.44,0.2),size=9, 
             color='navy',alpha=0)
t2.set_bbox(dict(facecolor='navy', alpha=0.13, edgecolor='navy'))
plt.annotate('>',xy=(0.6,0.18),size=18, 
             color='k')
t = plt.annotate('AUC',xy=(0.71,0.2),size=9,
             color='navy')
t.set_bbox(dict(facecolor="None",edgecolor='navy',linestyle='--'))

# draw arrow operating point
plt.annotate('operating point',
xy=(OP[0], OP[1]), 
arrowprops=dict(arrowstyle='->',
                connectionstyle="angle3"), 
xytext=(0.03, 0.91),size=9)

# draw arrow to connect ROC with panel A
plt.annotate('',
xy=(0.65,0.98), 
arrowprops=dict(arrowstyle='->',
                connectionstyle="angle3,angleA=20,angleB=40"), 
xytext=(0.9, 1.14),size=9)
        
# plot labels and arange positions
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# plot panel label C
ax4.text(0.05, 1.09, 'C', transform=ax4.transAxes,
      fontsize=16, fontweight='bold', va='top', ha='right')

# adjust whitespace
fig.subplots_adjust(top = 0.96, bottom = 0.08, right = 0.99, left = 0.18, 
        hspace = 0.15, wspace = 0.3)
# save figure
plt.savefig('../Figures/ExhaustiveFeatureSelection.png',dpi=600)
plt.show()
