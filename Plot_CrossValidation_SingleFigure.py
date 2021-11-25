# import packages
import matplotlib.pyplot as plt
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
# define figure
fig, ax = plt.subplots(figsize=(8.8/2.54, 8.8/2.54))

#  loop through all field sites
for j in range(len(FSs)):
    # read in ROC of a specific field site
    ROC = pd.read_csv('../Results/'+combinations_slope_stemp[0]+
                                  combinations_slope_stemp[1]+
                                  'SpecNeg/ROC_values_CrossValidation_'+
                      FSs[j]+'_'+version+'.csv')
    # plot ROC of a specific field site
    plt.plot(ROC.FPrate2,ROC.TPrate2,label=FSs[j],
             color=colors[j],linestyle=linestyles[j])

# read in average ROC
ROC_average = pd.read_csv('../Results/'+combinations_slope_stemp[0]+
                                  combinations_slope_stemp[1]+
                                  'SpecNeg/ROC_average_'
                          +version+'.csv')
# plot average ROC
plt.plot(ROC_average.FalsePositive_average,ROC_average.TruePositive_average,
         label='weighted average',color='k')

# plot diagonal line
plt.plot([0,1],[0,1],alpha=1, linewidth=0.1,color='k')

# plot settings
plt.gca().set_aspect('equal', adjustable='box')
box = ax.get_position()
ax.set_position([box.x0, box.y0, 0.5 * 0.9, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().subplots_adjust(right=0.7)
plt.title('ROC curves Cross Validation')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
#plt.savefig('../Figures/ROCs_CrossValidation'+version+'.png',dpi=200)

