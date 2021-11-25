# import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# define total number of positive classified observations, negative classified observations and meteorite finds
total_positives = 106687
total_negatives = 1823002
n_mets_today = 45213

#%%
# define function that calculates the percentage of meteorites found today
def n_posobs(total_positives,
             precision,
             sensitivity,
             n_mets_per_cell,
             n_mets_today):
    TP_t = precision * total_positives
    FN_t = (1-sensitivity)/sensitivity * TP_t
    n_posobs = TP_t + FN_t
    n_mets = n_posobs*n_mets_per_cell
    perc_mets_found = (n_mets_today/n_mets)*100
    print('number of positive observations',np.round(n_posobs,0),
          '\n number of meteorites',np.round(n_mets,0),
          '\n percentage found',np.round(perc_mets_found,1))
    return(perc_mets_found)
# caluclate upper and lower boundary of percentage meteorites found today
n_max = n_posobs(total_positives,0.47,0.74,5,n_mets_today)
n_min = n_posobs(total_positives,0.81,0.48,5,n_mets_today)

#%% 
# plot relation between percentage of meteorites found, sensitivity and precision
# define interval of precision
precision = np.linspace(0.4,0.7,10)
# define interval of sensitivity
sensitivity = np.linspace(0.45,0.75,10)
# define number of meteorites per gridcell
n_mets_per_cell = 5
# create empty array to store percentage of meteorites found today
perc_mets_found = np.zeros((len(precision),len(sensitivity)))

# loop through precision and sensitivity values to calculate percentage of meteorites found for each combination
for i in range(len(precision)):
    for j in range(len(sensitivity)):
        TP_t = precision[i] * total_positives
        FN_t = (1-sensitivity[j])/sensitivity[j] * TP_t
        perc_mets_found[i,j] = 44000/((TP_t + FN_t)*n_mets_per_cell)*100

# generate 2d figure of percentage of meteorites found depending on precision and sensitivity
cbar = sns.heatmap(perc_mets_found,cmap="viridis",
                   cbar_kws={'label': 'percentage meteorites found'})
# plot labels
plt.ylabel('precision')
plt.xlabel('sensitivity')
# plot ticks
plt.yticks(np.linspace(0.5,len(precision)-0.5,len(precision)),
           np.round(precision,2),rotation=0)
plt.xticks(np.linspace(0.5,len(sensitivity)-0.5,len(sensitivity)),
           np.round(sensitivity,2),rotation=0)

plt.show()

