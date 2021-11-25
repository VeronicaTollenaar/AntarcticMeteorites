# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# define version of classification
version = '4pcs4fs0134'
# define version of surface temperature and slope
combinations_slope_stemp = ['slope2km','stempPERC99']

# define abbreviations of all field sies
FSs = ['QUE','MIL','LEW','EET','GRV','ALH','MAC','PCA','FRO','rest']

# define colors corresponding to different field sites (hardcoded)
colors = ['#4477aa','#4477aa','#4477aa',
          '#66ccee',
          '#228833',
          '#66ccee',
          '#4477aa',
          '#ccbb44',
          '#ee6677',
          '#aa3377']
# define linestyles for different field sites (hardcoded)
linestyles = ['solid',
              'solid',
              (0, (4, 1)),
              'solid',
              'solid',
              'solid',
              (0, (1, 1, 1, 1)),
              'solid',
              'solid',
              'solid']

# read in data average ROC curve
ROC_average = pd.read_csv('../Results/'+combinations_slope_stemp[0]+
                                  combinations_slope_stemp[1]+
                                  'SpecNeg/ROC_average_'
                          +version+'.csv')
# set font
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

# loop through all field sites
for j in range(len(FSs)):
    # define figure
    fig, ax = plt.subplots(figsize=(2.5/2.54, 2.5/2.54))
    # plot average ROC
    plt.plot(ROC_average.FalsePositive_average,ROC_average.TruePositive_average,
         label='weighted average',color='k',linewidth=0.8)
    # read in ROC of specific field site
    ROC = pd.read_csv('../Results/'+combinations_slope_stemp[0]+
                                  combinations_slope_stemp[1]+
                                  'SpecNeg/ROC_values_CrossValidation_'+
                      FSs[j]+'_'+version+'.csv')
    # plot ROC of specific field site
    plt.plot(ROC.FPrate2,ROC.TPrate2,label=FSs[j],
             color=colors[j],linestyle=linestyles[j])
    
    # plot diagonal line (not visible)
    plt.plot([0,1],[0,1],alpha=1, linewidth=0,color='k')
    # plot settings
    plt.gca().set_aspect('equal', adjustable='box')
    box = ax.get_position()
    # plot labels
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    # adjust spacing labels
    ax.xaxis.labelpad = -8.5
    ax.yaxis.labelpad = -8.5
    # adjust ticks
    plt.yticks(np.arange(0, 2, step=1))
    plt.xticks(np.arange(0, 2, step=1))
    # adjust spacing figure
    fig.subplots_adjust(top = 0.99, bottom = 0.21, right = 0.99, left = 0.20, 
            hspace = 0.8, wspace = 0.8)
    # save figure
    # fig.savefig('../Figures/ROCs_CrossValidation'+version+combinations_slope_stemp[0]+
    #                                   combinations_slope_stemp[1]+FSs[j]+'.png',dpi=400)
    plt.show()
    

# 3 in 1 plot for QUE,LEW,MAC
# define figures
fig, ax = plt.subplots(figsize=(2.5/2.54, 2.5/2.54))
# loop through field sites
for i in range(len(FSs)):
    # select field sites QUE, LEW, and MAC
    if FSs[i] == 'QUE' or FSs[i] == 'LEW' or FSs[i] == 'MAC':
        # plot average ROC
        plt.plot(ROC_average.FalsePositive_average,ROC_average.TruePositive_average,
              color='k',linewidth=0.8)
        # read in ROC of specific field site
        ROC = pd.read_csv('../Results/'+combinations_slope_stemp[0]+
                                      combinations_slope_stemp[1]+
                                      'SpecNeg/ROC_values_CrossValidation_'+
                          FSs[i]+'_'+version+'.csv')
        # plot ROC of specific field site
        plt.plot(ROC.FPrate2,ROC.TPrate2,label=FSs[i],
                  color=colors[i],linestyle=linestyles[i])
        
# plot diagonal line (not visible)
plt.plot([0,1],[0,1],alpha=1, linewidth=0,color='k')
# plot settings
plt.gca().set_aspect('equal', adjustable='box')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, 0.5 * 0.9])

# plot labels
plt.xlabel('FP rate')
plt.ylabel('TP rate')
# adjust spacing labels
ax.xaxis.labelpad = -8.5
ax.yaxis.labelpad = -8.5
# adjust ticks
plt.yticks(np.arange(0, 2, step=1))
plt.xticks(np.arange(0, 2, step=1))
# adjust spacing figure
fig.subplots_adjust(top = 0.99, bottom = 0.21, right = 0.99, left = 0.20, 
            hspace = 0.8, wspace = 0.8)
# save figure
# fig.savefig('../Figures/ROCs_CrossValidation'+version+combinations_slope_stemp[0]+
#                                       combinations_slope_stemp[1]+'QUELEWMAC.png',dpi=400)



