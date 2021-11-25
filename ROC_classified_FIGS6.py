# import packages
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# import ultimate test neg
test_neg_all = pd.read_csv('../Data_Locations/TEST_neg.csv')
# exclude negative test data that is in negative validation data
validation_neg = pd.read_csv(
    '../Data_Locations/validation_neg.csv')
validation_neg['in_validation'] = 1
test_neg_merged = test_neg_all.merge(validation_neg,
                                 how='outer',
                                 on =['x','y'])
test_neg = test_neg_merged[
                 np.isnan(test_neg_merged.in_validation)].drop(
                 ['in_validation'],
                 axis=1)[:][['x','y']]

# for random test data: read in radar backscatter values (arbitrary feature)
f1_radar_toclass = pd.read_csv(
    '../Data_Features/radarbackscatter_at_toclass.csv').drop(['radar'],axis=1)
# select a random sample
test_rand = f1_radar_toclass.sample(4000,random_state=11)
# ensure random test data is not in negative validation data
validation_rand = pd.read_csv(
    '../Data_Locations/validation_neg.csv')
validation_rand['in_validation'] = 1
test_rand_merged = test_rand.merge(validation_rand,
                                 how='outer',
                                 on =['x','y'])
test_rand = test_rand_merged[
                 np.isnan(test_rand_merged.in_validation)].drop(
                 ['in_validation'],
                 axis=1)[:][['x','y']]
print(len(test_rand))

# import ultimate test pos
test_pos_all = pd.read_csv('../Data_Locations/TEST_pos.csv')
# exclude positive test data that is in positive validation data
validation_pos = pd.read_csv(
    '../Data_Locations/locations_mets.csv')[[
        'x','y']]
validation_pos['in_validation'] = 1
test_pos_merged = test_pos_all.merge(validation_pos,
                                 how='outer',
                                 on =['x','y'])
test_pos = test_pos_merged[
                 np.isnan(test_pos_merged.in_validation)].drop(
                 ['in_validation'],
                 axis=1)[:]          

del(test_pos_all,test_pos_merged,test_rand_merged,
    validation_pos,validation_neg,test_neg_merged,
    test_neg_all,validation_rand)
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
# open unlabelled data
f1_radar_toclass = pd.read_csv('../Data_Features/radarbackscatter_at_toclass.csv')
f2_speed_toclass = pd.read_csv('../Data_Features/velocities_at_toclass.csv')
f4_slope_toclass = pd.read_csv('../Data_Features/slope2km_at_toclass.csv')
f5_stemp_toclass = pd.read_csv('../Data_Features/stempPERC99_at_toclass.csv')

# merge data
data_toclass = f1_radar_toclass.merge(
                f2_speed_toclass).merge(
                f4_slope_toclass).merge(
                f5_stemp_toclass)


# delete individual features
del(f1_radar_toclass,
    f2_speed_toclass,
    f4_slope_toclass,
    f5_stemp_toclass)
                   
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

#%%
# reorganize the data into groups of:
# - labelled training data
# - unlabelled training data
# - all data
# - test data

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

# select test data from all data
test_pos_sel = train_all.merge(test_pos,how='inner',on=['x','y'])
test_neg_sel = train_all.merge(test_neg,how='inner',on=['x','y'])
test_rand_sel = train_all.merge(test_rand,how='inner',on=['x','y'])

del(data_mets_transf,
    lab_to_exclude,
    lab_to_exclude_merged,
    data_toclass_transf)
#%%
# standardize features
# use index [:,2:] as the first two columns are the x and y coordinates
scaler_train   = preprocessing.StandardScaler().fit(train_all.iloc[:,2:].values)
train_lab_st   = scaler_train.transform(train_lab.iloc[:,2:].values)
train_unlab_st = scaler_train.transform(train_unlab.iloc[:,2:].values)
train_all_st   = scaler_train.transform(train_all.iloc[:,2:].values)
test_pos_st    = scaler_train.transform(test_pos_sel.iloc[:,2:].values)
test_neg_st    = scaler_train.transform(test_neg_sel.iloc[:,2:].values)
test_rand_st   = scaler_train.transform(test_rand_sel.iloc[:,2:].values)

del(train_lab,train_unlab,scaler_train)

#%%
# compute principal components
pca = PCA()
pca.fit(train_all_st)

# transform standardized features to principal components
pcs_train_lab = pca.transform(train_lab_st)
pcs_train_unlab = pca.transform(train_unlab_st)
pcs_train_all = pca.transform(train_all_st)
pcs_test_pos = pca.transform(test_pos_st)
pcs_test_neg = pca.transform(test_neg_st)
pcs_test_rand = pca.transform(test_rand_st)

del(train_lab_st,train_unlab_st,train_all_st,pca)

#%%
# kernel density estimation labelled data
pcmax = 4 # number of principal components

# reorganize data
xy_train_lab  = np.array(pcs_train_lab[:,0:pcmax].tolist()).squeeze()

# define possible bandwiths
bandwidths = np.linspace(0.3,0.35,30)

# perform gridsearch (cross-validation) to estimate optimal bandwith
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv = 10)
grid.fit(xy_train_lab);

# print bandwidth to check it is within the limits of the bandwiths
print(grid.best_params_)
bw_lab = grid.best_params_['bandwidth']

# estimate density using the findal bandwidth
kde_lab = KernelDensity(bandwidth=bw_lab).fit(xy_train_lab)

#%%
# score all data using kde_lab
# reorganize data
xy_test_pos = np.array(pcs_test_pos[:,0:pcmax].tolist()).squeeze()
xy_test_neg = np.array(pcs_test_neg[:,0:pcmax].tolist()).squeeze()
xy_test_rand= np.array(pcs_test_rand[:,0:pcmax].tolist()).squeeze()
# score data
scores_test_pos_lab = np.exp(kde_lab.score_samples(xy_test_pos))
scores_test_neg_lab = np.exp(kde_lab.score_samples(xy_test_neg))
scores_test_rand_lab = np.exp(kde_lab.score_samples(xy_test_rand))

#%%
# kernel density estimation unlabelled data
# select a random sample of 10.000 from the unlabelled data
pcs_train_unlab_df = pd.DataFrame(pcs_train_unlab)
randsample_pcs_train_unlab = pcs_train_unlab_df.sample(
                             10000,random_state=5).values
# reorganize data
xy_train_unlab  = np.array(randsample_pcs_train_unlab[:,0:pcmax].tolist()).squeeze()

# define possible bandwiths
bandwidths = np.linspace(0.2,0.3,30)

# perform gridsearch (cross-validation) to estimate optimal bandwith
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv = 10)
grid.fit(xy_train_unlab);

# print bandwidth to check it is within the limits of the bandwiths
print(grid.best_params_)
bw_unlab = grid.best_params_['bandwidth']

# estimate density using the findal bandwidth
kde_unlab = KernelDensity(bandwidth=bw_unlab).fit(xy_train_unlab)

#%%
# score all data using kde_unlab
# score data
scores_test_pos_unlab = np.exp(kde_unlab.score_samples(xy_test_pos))
scores_test_neg_unlab = np.exp(kde_unlab.score_samples(xy_test_neg))
scores_test_rand_unlab = np.exp(kde_unlab.score_samples(xy_test_rand))

#%%
# estimate following probabilities:
    
# p(x|s=1) = scored_lab
# p(x|s=0) = scored_unlab 
# p(s=1)
# p(s=0)
# p(s=1|x)
# p(x=0|x)

# define possible values for cost parameter labda
cost = np.logspace(np.log10(0.1),np.log10(6000000),1000)

TPrateTest = np.zeros(len(cost))
FPrateTest = np.zeros(len(cost))
FPrateTest_rand = np.zeros(len(cost))

for k in range(len(cost)):
    ps1 = len(pcs_train_lab)*cost[k]/(len(pcs_train_lab)*cost[k] + 
                                       len(pcs_train_unlab))
    ps0 = len(pcs_train_unlab)/(len(pcs_train_lab)*cost[k] + 
                                       len(pcs_train_unlab))
    
    def probs(inp_scores_lab,inp_scores_unlab):
        ps1gx = (inp_scores_lab*ps1)/(inp_scores_lab*ps1 + inp_scores_unlab*ps0)
        ps0gx = (inp_scores_unlab*ps0)/(inp_scores_lab*ps1 + inp_scores_unlab*ps0)
        return ps1gx, ps0gx
    
    ps1gx_test_pos, ps0gx_test_pos = probs(scores_test_pos_lab,scores_test_pos_unlab)
    ps1gx_test_neg, ps0gx_test_neg = probs(scores_test_neg_lab,scores_test_neg_unlab)
    ps1gx_test_rand, ps0gx_test_rand = probs(scores_test_rand_lab,scores_test_rand_unlab)

    TPrateTest[k] = len(ps1gx_test_pos[(ps1gx_test_pos)>0.5])/len(test_pos_sel)
    FPrateTest[k] = len(ps1gx_test_neg[(ps1gx_test_neg)>0.5])/len(test_neg_sel)
    FPrateTest_rand[k] = len(ps1gx_test_rand[(ps1gx_test_rand)>0.5])/len(test_rand_sel)
#print(len(positive_classified))

#%%
# export restults as ROC curves
ROC_test_neg_df = pd.DataFrame({'cost': cost,
                            'TPrateTest': TPrateTest,
                            'FPrateTest': FPrateTest})
ROC_test_rand_df = pd.DataFrame({'cost': cost,
                            'TPrateTest': TPrateTest,
                            'FPrateTest': FPrateTest_rand})
ROC_test_neg_df.to_csv('../Results/ROC_Testdata_'+version_tosave+'.csv',
                            index=False)
ROC_test_rand_df.to_csv('../Results/ROC_Testdata_'+version_tosave+'_Randneg.csv',
                            index=False)

#%%
# plot ROC curves obtained with test data and validation data
# read in data
ROC_val_rand = pd.read_csv('../Results/slope2kmstempPERC99RandNeg/ROC_average_4pcs4fs0134.csv')
ROC_test_rand = pd.read_csv('../Results/ROC_Testdata_slope2kmstempPERC99_4pcs4fs0134_Randneg.csv')

ROC_val_spec = pd.read_csv('../Results/slope2kmstempPERC99SpecNeg/ROC_average_4pcs4fs0134.csv')
ROC_test_spec = pd.read_csv('../Results/ROC_Testdata_slope2kmstempPERC99_4pcs4fs0134.csv')

# set font
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
# define figure
fig, ax1 = plt.subplots(figsize=(8.8/2.54, 8.8/2.54))
# plot data
plt.plot(ROC_val_spec.FalsePositive_average,
         ROC_val_spec.TruePositive_average,label='calibration, negative data',
         color = 'navy',
         linestyle='--',
         zorder=0)
plt.plot(ROC_test_spec.FPrateTest,
         ROC_test_spec.TPrateTest,label='test, negative data',
         color = 'navy',
         zorder=0)

plt.scatter(ROC_val_spec[ROC_val_spec.cost==cost_sel[0]].FalsePositive_average,
         ROC_val_spec[ROC_val_spec.cost==cost_sel[0]].TruePositive_average,
         label='selected operating point',
         color = 'k',
         s=25,
         marker='x',
         zorder=2)
plt.scatter(ROC_test_spec[ROC_test_spec.cost==cost_sel[0]].FPrateTest,
          ROC_test_spec[ROC_test_spec.cost==cost_sel[0]].TPrateTest,
          color = 'k',
          s=25,
          marker='x',
          zorder=2)

plt.plot(ROC_val_rand.FalsePositive_average,
         ROC_val_rand.TruePositive_average,label='calibration, random data',
         color = 'cornflowerblue',
            linestyle = '--',
            zorder=0)
plt.plot(ROC_test_rand.FPrateTest,
         ROC_test_rand.TPrateTest,label='test, random data',
         color = 'cornflowerblue',
            linestyle = '-',
            zorder=0)

plt.scatter(ROC_val_rand[ROC_val_rand.cost==cost_sel[0]].FalsePositive_average,
            ROC_val_rand[ROC_val_rand.cost==cost_sel[0]].TruePositive_average,
            color = 'k',
            s=25,
            marker='x',
            zorder=2)
plt.scatter(ROC_test_rand[ROC_test_rand.cost==cost_sel[0]].FPrateTest,
            ROC_test_rand[ROC_test_rand.cost==cost_sel[0]].TPrateTest,
            color = 'k',
            s=25,
            marker='x',
            zorder=2)

# adjust number of x and y labels
plt.locator_params(nbins=6)
plt.legend(loc='lower right')
plt.gca().set_aspect('equal', 'box')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curves Calibration and Test data')
# adjust whitespace and save figure
fig.subplots_adjust(top = 0.93, bottom = 0.12, right = 01.4, left = -0.25, 
        hspace = 0.15, wspace = 0.3)
#plt.savefig('../Figures/ROCcurves_val_and_test.png',dpi=300)
#%%
# function to calculate the AUC
def auc(ROC_input,colnameFP,colnameTP):
    ROC_added = pd.concat([pd.DataFrame([
                    {colnameFP: 0,
                     colnameTP: 0}]),
                     ROC_input,
                    pd.DataFrame([
                    {colnameFP: 1,
                     colnameTP: 1}])]) 
    aucs = metrics.auc(ROC_added[colnameFP],
                          ROC_added[colnameTP])
    return aucs
# calculate the AUC for the four different curves
auc_test_spec = auc(ROC_test_spec,'FPrateTest','TPrateTest')
auc_test_rand = auc(ROC_test_rand,'FPrateTest','TPrateTest')
auc_val_spec = auc(ROC_val_spec,'FalsePositive_average','TruePositive_average')
auc_val_rand = auc(ROC_val_rand,'FalsePositive_average','TruePositive_average')
# print values
print(auc_val_spec,
      auc_val_rand,
      auc_test_spec,
      auc_test_rand
      )
# print ratios
print(auc_test_spec/auc_val_spec)
print(auc_test_rand/auc_val_rand)
