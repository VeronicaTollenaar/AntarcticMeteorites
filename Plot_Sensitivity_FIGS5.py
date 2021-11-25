# import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# font settings
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
# define figure
fig, ax = plt.subplots(2,1,figsize=(18/2.54, 12/2.54))

## plot upper panel
color ='navy'

# define area under the curve of the original classification
auc_original_specneg = 0.8318854610186937

# read in and plot data per sensitivity analysis
# values to plot consist of the AUC divided over the AUC of the original classification
#"trainingdata"
aucs_train = pd.read_csv('../Results/SensitivityTrainingdataSpecNeg.csv')
negative = ax[0].bar(0.64 + aucs_train['Unnamed: 0']/12.5, aucs_train.AUC/auc_original_specneg*100,
        width=0.07,color=color)

#"density"
aucs_density = pd.read_csv('../Results/SensitivityDensitySpecNeg.csv')
ax[0].bar(1.8,aucs_density['0']/auc_original_specneg*100,color=color,
        width = 0.4)

#"isolated"
aucs_isolated = pd.read_csv('../Results/SensitivityIsolatedSpecNeg.csv')
ax[0].bar(2.4,aucs_isolated['0']/auc_original_specneg*100,color=color,
        width = 0.4)

#"wind displacement"
aucs_wind100 = pd.read_csv('../Results/SensitivityWind100gSpecNeg.csv')
ax[0].bar(2.925,aucs_wind100['0']/auc_original_specneg*100,
        width=0.25,color=color)
aucs_wind150 = pd.read_csv('../Results/SensitivityWind150gSpecNeg.csv')
ax[0].bar(3.2,aucs_wind150['0']/auc_original_specneg*100,
        width=0.25,color=color)
aucs_wind200 = pd.read_csv('../Results/SensitivityWind200gSpecNeg.csv')
ax[0].bar(3.475,aucs_wind200['0']/auc_original_specneg*100,
        width=0.25,color=color)

# plot horizontal line at 100%
xmin = 0.5
xmax = 3.7
ax[0].hlines(100,xmin,xmax,linestyles='-',color='k',linewidth=0.5)

# plot settings
ax[0].set_xlim([xmin,xmax])
plt.ylim([0,110])
fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax[0].yaxis.set_major_formatter(yticks)
ax[0].xaxis.tick_top()

# plot labels
ax[0].set_ylabel('AUC (%, vs. original AUC)')
ax[0].set_xticks((1,1.8,2.4,3.2))
ax[0].set_xticklabels(['w/o fieldsite',
                      'exact locations',
                      'w/o isolated finds',
                      'w/o light meteorites'])


## plot lower panel
color ='cornflowerblue'

# define area under the curve of the original classification
auc_original_randneg = 0.9558971357118525

# read in and plot data per sensitivity analysis
# values to plot consist of the AUC divided over the AUC of the original classification
#"trainingdata"
aucs_train = pd.read_csv('../Results/SensitivityTrainingdataRandNeg.csv')
random = ax[1].bar(0.64 + aucs_train['Unnamed: 0']/12.5, 
        aucs_train.AUC/auc_original_randneg*100,
        width=0.07,color=color)

#"density"
aucs_density = pd.read_csv('../Results/SensitivityDensityRandNeg.csv')
ax[1].bar(1.8,aucs_density['0']/auc_original_randneg*100,color=color,
        width = 0.4)

#"isolated"
aucs_isolated = pd.read_csv('../Results/SensitivityIsolatedRandNeg.csv')
ax[1].bar(2.4,aucs_isolated['0']/auc_original_randneg*100,color=color,
        width=0.4)

#"wind displacement"
aucs_wind100 = pd.read_csv('../Results/SensitivityWind100gRandNeg.csv')
ax[1].bar(2.925,aucs_wind100['0']/auc_original_randneg*100,
        width=0.25,color=color)
aucs_wind150 = pd.read_csv('../Results/SensitivityWind150gRandneg.csv')
ax[1].bar(3.2,aucs_wind150['0']/auc_original_randneg*100,
        width=0.25,color=color)
aucs_wind200 = pd.read_csv('../Results/SensitivityWind200gRandNeg.csv')
ax[1].bar(3.475,aucs_wind200['0']/auc_original_randneg*100,
        width=0.25,color=color)

# plot horizontal line at 100%
ax[1].hlines(100,xmin,xmax,linestyles='-',color='k',linewidth=0.5)

# plot settings
ax[1].set_xlim([xmin,xmax])
plt.ylim([0,110])
fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax[1].yaxis.set_major_formatter(yticks)
plt.xlim([xmin,xmax])

# plot labels
plt.ylabel('AUC (%, vs. original AUC)')
plt.xticks((0.64 + aucs_train['Unnamed: 0']/12.5).append(
    pd.Series([2.925, 3.2, 3.475])),
            aucs_train.min_fieldsite.append(
                pd.Series(['100','150','200'])),rotation=90)
plt.xlabel('fieldsite')
ax[1].xaxis.set_label_coords(0.14, -0.325)
ax3 = ax[1].twiny()
ax3.xaxis.tick_bottom()
plt.xticks([])
plt.xlabel('threshold (< gram)')
ax3.xaxis.set_label_coords(0.86, -0.392)

# plot legend
plt.legend((negative, random),['negative data','random data'],bbox_to_anchor=(0.62, -0.04, 0, 0))

# plot title
plt.suptitle('Sensitivity to change in positive training data')
# adjust spacing and save figure
fig.subplots_adjust(top = 0.86, bottom = 0.14, right = 0.995, left = 0.09, 
        hspace = 0.2, wspace = 0.3)
#plt.savefig('../Figures/Sensitivity.png',dpi=300)
plt.show()
