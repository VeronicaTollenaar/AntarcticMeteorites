# import packages
import matplotlib.pyplot as plt
import os
import pandas as pd


# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# set threshold/select operating point on ROC curve by maximizing the F-1 score

# open cross validation data obtained in feature selection
version = 'slope2kmstempPERC99SpecNeg/ROC_average_4pcs4fs0134'
version_tosave = 'slope2kmstempPERC99_4pcs4fs0134'
ROC_average = pd.read_csv('../Results/'+version+'.csv')

# calculate number of True Positives (TP), True Negatives (TN), 
# False Positives (FP) and False Negatives (FN)
n_pos_obs_CV = 2541 # number of positive observations used in cross validation
n_neg_obs_CV = 8726 # number of negative observations used in cross validation
TP = ROC_average.TruePositive_average*n_pos_obs_CV
TN = (1-ROC_average.FalsePositive_average)*n_neg_obs_CV
FP = ROC_average.FalsePositive_average*n_neg_obs_CV
FN = (1-ROC_average.TruePositive_average)*n_pos_obs_CV

# calculate Accuracy (ACC), Precision (PREC), Recall (REC) and F1-score (F1)
ACC = (TP+TN)/(TP+TN+FP+FN)
PREC = TP/(TP+FP)
REC = TP/(TP+FN)
F1 = 2*TP/(2*TP + FP + FN)
SPEC = TN/(TN+FP)

# plot performance measures versus cost
cost = ROC_average.cost
imax=480
plt.plot(cost[:imax],ACC[:imax],label='Accuracy')
plt.plot(cost[:imax],PREC[:imax],label='Precision')
plt.plot(cost[:imax],REC[:imax],label='Recall')
plt.plot(cost[:imax],SPEC[:imax],label='Specificity')
plt.plot(cost[:imax],F1[:imax],label='F1 Score',color='k')
# scatter maximum of F1 value
plt.scatter(cost[F1==F1.max()],F1.max(),color='k',marker='x')
# plot labels, title, legend
plt.xlabel('cost')
plt.ylabel('fraction')
plt.title('Precision, Recall, and F1 Score')
plt.legend()
#plt.savefig('../Figures/PerformanceMeasures_vs_cost.png',dpi=200)
plt.show()

cost_sel = ROC_average[F1==F1.max()].cost.values
cost_idx = F1.idxmax(axis=0)

print('accuracy:', ACC[cost_idx])
print('precision:', PREC[cost_idx])
print('F1 score:', F1[cost_idx])
print('recall', REC[cost_idx])
print('specificity', SPEC[cost_idx])
print(TP[cost_idx],FN[cost_idx],FP[cost_idx],TN[cost_idx])

#%% 
# open cross validation data RANDNEG
version = 'slope2kmstempPERC99RandNeg/ROC_average_4pcs4fs0134'
ROC_average = pd.read_csv('../Results/'+version+'.csv')
# calculate number of True Positives (TP), True Negatives (TN), 
# False Positives (FP) and False Negatives (FN)
TP = ROC_average.TruePositive_average*2541
TN = (1-ROC_average.FalsePositive_average)*9000
FP = ROC_average.FalsePositive_average*9000
FN = (1-ROC_average.TruePositive_average)*2541
# calculate Accuracy (ACC), Precision (PREC), Recall (REC) and F1-score (F1)
ACC = (TP+TN)/(TP+TN+FP+FN)
PREC = TP/(TP+FP)
REC = TP/(TP+FN)
F1 = 2*TP/(2*TP + FP + FN)
SPEC = TN/(TN+FP)

print('accuracy:', ACC[cost_idx])
print('precision:', PREC[cost_idx])
print('F1 score:', F1[cost_idx])
print('recall:', REC[cost_idx])
print('specificity:', SPEC[cost_idx])
print(TP[cost_idx],FN[cost_idx],FP[cost_idx],TN[cost_idx])










