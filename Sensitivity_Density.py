# import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
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

# outline classification:
# - read in all data
# - PCA
# - KDE on positive observations
# - KDE on unlabelled observations
# - evaluation of KDEs on unlabelled observations
# - classify observation as MSZ or non-MSZ

#%%
# open labelled data (REPROJECTED LOCATIONS)
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
# open labelled data (EXACT LOCATIONS)
f1_radar_mets = pd.read_csv('../Data_Features/radarbackscatter_exactlocs_at_mets.csv')
f2_speed_mets = pd.read_csv('../Data_Features/velocities_exactlocs_at_mets.csv')
f4_slope_mets = pd.read_csv('../Data_Features/slope2km_exactlocs_at_mets.csv')
f5_stemp_mets = pd.read_csv('../Data_Features/stempPERC99_exactlocs_at_mets.csv')

# merge data
data_mets_EXACT = f1_radar_mets.merge(
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
data_mets_EXACT_transf = data_mets_EXACT.copy()
data_toclass_transf = data_toclass.copy()

data_mets_transf['radar']    = data_mets.radar + np.random.RandomState(
                               3).normal(0,0.25,len(data_mets.radar))
data_mets_EXACT_transf['radar']    = data_mets_EXACT.radar + np.random.RandomState(
                               4).normal(0,0.25,len(data_mets_EXACT.radar))
data_toclass_transf['radar'] = data_toclass.radar + np.random.RandomState(
                               6).normal(0,0.25,len(data_toclass.radar))

data_mets_transf['speed']    = np.log10(data_mets.speed.values)
data_mets_EXACT_transf['speed']    = np.log10(data_mets_EXACT.speed.values)
data_toclass_transf['speed'] = np.log10(data_toclass.speed.values)

data_mets_transf['slope_max']    = np.log10(data_mets.slope_max.values)
data_mets_EXACT_transf['slope_max'] = np.log10(data_mets_EXACT.slope_max.values)
data_toclass_transf['slope_max'] = np.log10(data_toclass.slope_max.values)

data_mets_transf['stemp']    = data_mets.stemp + np.random.RandomState(
                               5).normal(0,0.04,len(data_mets.stemp))
data_mets_EXACT_transf['stemp']    = data_mets_EXACT.stemp + np.random.RandomState(
                               7).normal(0,0.04,len(data_mets_EXACT.stemp))
data_toclass_transf['stemp'] = data_toclass.stemp + np.random.RandomState(
                               8).normal(0,0.04,len(data_toclass.stemp))#%%
#%%
# read in abbreviations of meteorite recovery locations (REPROJECTED)
locs_mets = pd.read_csv(
        '../Data_Locations/locations_mets_abbrevs.csv')[[
        'x','y','abbrevs','counts']]
data_mets_locs = data_mets_transf.merge(locs_mets)
#names of 9 largest fieldsites
FSs = ['QUE','MIL','LEW','EET','GRV','ALH','MAC','PCA','FRO','rest']
#%%
# read in abbreviations of meteorite recovery locations (EXACT)
locs_mets_EXACT = pd.read_csv(
        '../Data_Locations/locations_mets_nogrid.csv')[[
        'x','y','abbrevs']] #mttb_locs_adj
locs_mets_EXACT = locs_mets_EXACT.rename(columns={'new_x':'x','new_y':'y'})
data_mets_locs_EXACT = data_mets_EXACT_transf.merge(locs_mets_EXACT)

#%%
# define function that performs cross validation for given set of observations
def crossvalroc(data_mets, # positive observations
                data_mets_EXACT, # positive observations (EXACT FINDING LOCATIONS)
                data_toclass, # unlabelled observations
                cost, # array of different values of cost parameter labda
                version, # name of the version (used to save the data)
                pcmax, # number of principal components
                neg # type of negative validation data (specific (negative) or random)
                ):
    
    # merge positive labeled data with all data
    posshp = data_mets.copy()[['x','y']]
    posshp['pos'] = 1
    merged_all_pos = data_toclass.merge(posshp, how='outer',
                                      on =['x','y'])    
    
    # define specific negative validation data ("test_neg") for two scenarios:
    # 1. negative validation data (SpecNeg), 2. random validation data (RandNeg)
    if neg == 'SpecNeg':
        specific_neg = pd.read_csv('../Data_Locations/validation_neg.csv')
        # ensure negative validation data is subset of unlabelled data
        # exclude positive observations from negative validation data
        merged_specific_neg = merged_all_pos.merge(specific_neg, how='outer')
        merged_specific_neg = merged_specific_neg.rename(columns={"bias": "neg"})
        test_specific_neg = merged_specific_neg[(merged_specific_neg.neg==1)
                                     &(np.isnan(merged_specific_neg.pos))
                                     &(pd.notnull(merged_specific_neg.iloc[:,2]))
                                     ].drop(['neg','pos'],axis=1)
        test_neg = test_specific_neg
        merged_all_test = merged_specific_neg
        print('len SpecNeg', len(test_neg))

    # define random negative validation data
    if neg == 'RandNeg':
        data_toclass_df = pd.DataFrame(merged_all_pos)
        # exclude positive observations from all observations
        data_toclass_nopos = data_toclass_df[
                            (np.isnan(merged_all_pos.pos))
                            ].drop(['pos'],axis=1)
        # sample random validation data
        random_neg = data_toclass_nopos.sample(9000,random_state=10)
        test_random_neg = random_neg.copy()
        random_neg['neg']=1
        test_neg = test_random_neg
        merged_all_test = merged_all_pos.merge(random_neg,how='outer')
        print('len RandNeg', len(test_neg)) 
   
    # exclude negative validation data from train data
    train_unlab_pre = merged_all_test[(np.isnan(merged_all_test.neg))
                                          ].drop(['neg'],axis=1)
    # exclude positive labeled data from unlabeled data
    train_unlab = train_unlab_pre[np.isnan(train_unlab_pre.pos)
                                          ].drop(['pos'],axis=1)  
    
    ## Standardize features
    # standardize features using unlabeled data (for computational efficiency)
    scaler_train = preprocessing.StandardScaler().fit(train_unlab.iloc[:,2:].values)
    train_unlab_st = scaler_train.transform(train_unlab.iloc[:,2:].values)
    # standardize features test_neg
    test_neg_st = scaler_train.transform(test_neg.iloc[:,2:].values)    

    ## Compute Principal Components
    pca = PCA()
    pca.fit(train_unlab_st)
    # transpose unlabeled and negative data to principal components
    pcs_train_unlab_st = pca.transform(train_unlab_st)
    pcs_test_neg_st = pca.transform(test_neg_st)

    ## KDE on unlabeled observations
    # find best value for bandwidth with 10-fold cross validation
    # select random sample from pcs_train_unlab (computational efficiency)
    pcs_train_unlab_st_df = pd.DataFrame(pcs_train_unlab_st)
    randsample_pcs_train_unlab_st = pcs_train_unlab_st_df.sample(10000,random_state=5).values # SET BACK TO 10.000!!!!
    xy_train_unlab  = np.array(randsample_pcs_train_unlab_st[:,0:pcmax].tolist()).squeeze()
    bandwidths = np.linspace(0.1,0.6,30)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv = 10)
    grid.fit(xy_train_unlab);
    print('bw unlab:',grid.best_params_)
    bw_unlab = grid.best_params_['bandwidth']
    kde_unlab = KernelDensity(bandwidth=bw_unlab).fit(xy_train_unlab)

    # evaluate KDE
    xy_test_neg  = np.array(pcs_test_neg_st[:,0:pcmax].tolist()).squeeze()
    testscores_neg_unlab = np.exp(kde_unlab.score_samples(xy_test_neg))
  
    # for every set of positive test data
    for b in range(0,10): #range(len(FSs)) 
        # define positive validation data (test_pos)
        if FSs[b]=='rest':
            test_pos = data_mets[(data_mets.abbrevs!='QUE') &
                              (data_mets.abbrevs!='MIL') &
                              (data_mets.abbrevs!='LEW') &
                              (data_mets.abbrevs!='EET') &
                              (data_mets.abbrevs!='GRV') &
                              (data_mets.abbrevs!='ALH') &
                              (data_mets.abbrevs!='MAC') &
                              (data_mets.abbrevs!='PCA') &
                              (data_mets.abbrevs!='FRO') ].iloc[:,:-2]
        else:
            test_pos = data_mets[(data_mets.abbrevs==FSs[b])].iloc[:,:-2]
        #print('len test_pos' + FSs[b] +'is', len(test_pos))
        # define positive train data (EXACT LOCATIONS!)
        if FSs[b]=='rest':
            train_lab = data_mets_EXACT[(data_mets_EXACT.abbrevs=='QUE') |
                              (data_mets_EXACT.abbrevs=='MIL') |
                              (data_mets_EXACT.abbrevs=='LEW') |
                              (data_mets_EXACT.abbrevs=='EET') |
                              (data_mets_EXACT.abbrevs=='GRV') |
                              (data_mets_EXACT.abbrevs=='ALH') |
                              (data_mets_EXACT.abbrevs=='MAC') |
                              (data_mets_EXACT.abbrevs=='PCA') |
                              (data_mets_EXACT.abbrevs=='FRO')].iloc[:,:-1]
        else: 
            train_lab = data_mets_EXACT[(data_mets_EXACT.abbrevs!=FSs[b])].iloc[:,:-1]        
        print(len(train_lab))
        # standardize features train_lab
        train_lab_st = scaler_train.transform(train_lab.iloc[:,2:].values)
        # standardize features test_pos
        test_pos_st = scaler_train.transform(test_pos.iloc[:,2:].values)

        ## Compute Principal Components
        pcs_train_lab_st = pca.transform(train_lab_st)
        pcs_test_pos_st = pca.transform(test_pos_st)
        
        ## KDE on positive observations
        # find best value for bandwidth with 10-fold cross validation
        xy_train_lab  = np.array(pcs_train_lab_st[:,0:pcmax].tolist()).squeeze()
        bandwidths = np.linspace(0.1,0.5,20)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv = 10)
        grid.fit(xy_train_lab);
        print('bw lab:',grid.best_params_)
        bw_lab = grid.best_params_['bandwidth']
        kde_lab = KernelDensity(bandwidth=bw_lab).fit(xy_train_lab)

        # score validation data (testscores_neg_unlab is already done before the loop)
        xy_test_pos  = np.array(pcs_test_pos_st[:,0:pcmax].tolist()).squeeze()
        testscores_pos_lab = np.exp(kde_lab.score_samples(xy_test_pos))
        testscores_neg_lab = np.exp(kde_lab.score_samples(xy_test_neg))
        testscores_pos_unlab = np.exp(kde_unlab.score_samples(xy_test_pos))  

        ## calculate ROC curve
        # define empty arrays
        TPrate2 = np.zeros(len(cost))
        FPrate2 = np.zeros(len(cost))

        for i in range(len(cost)):
            # estimate following probabilities:
                # p(x|s=1) = scored_lab
                # p(x|s=0) = scored_unlab 
                # p(s=1)
                # p(s=0)
                # p(s=1|x)
                # p(x=0|x)
            ps1 = len(pcs_train_lab_st)*cost[i]/(len(pcs_train_lab_st)*cost[i]+
                                             len(pcs_train_unlab_st))
            ps0 = len(pcs_train_unlab_st)/(len(pcs_train_lab_st)*cost[i]+
                                       len(pcs_train_unlab_st))
            def probs(input_testscores_lab,input_testscores_unlab):
                ps1gx = (input_testscores_lab*ps1)/(input_testscores_lab*ps1 + input_testscores_unlab*ps0)
                ps0gx = (input_testscores_unlab*ps0)/(input_testscores_lab*ps1 + input_testscores_unlab*ps0)
                return ps1gx, ps0gx

            # calculate probabilities for validation data
            ps1gx_pos, ps0gx_pos = probs(testscores_pos_lab,testscores_pos_unlab)
            ps1gx_neg, ps0gx_neg = probs(testscores_neg_lab,testscores_neg_unlab)
            # calculate true positive and false positive rate
            TPrate2[i] = len(ps1gx_pos[(ps1gx_pos)>0.5])/len(ps1gx_pos)
            FPrate2[i] = len(ps1gx_neg[(ps1gx_neg)>0.5])/len(ps1gx_neg)
        
        # save values of ROC curve in folder
        ROCvals2 = pd.DataFrame({'cost': cost, 'TPrate2': TPrate2, 'FPrate2': FPrate2})
        ROCvals2.to_csv('../Results/ROC_values_CrossValidation_'+FSs[b]+'_'+version+'.csv',
                        index=False)        
    
#%%
# define possible values for cost parameter labda
cost = np.logspace(np.log10(0.1),np.log10(4000000),1000)
# perform cross validation using random calibration data
crossvalroc(data_mets_locs,
            data_mets_locs_EXACT,
            data_toclass_transf,
            cost,
            'SensitivityDensityRandNeg',
            4,
            'RandNeg')
# perform cross validation using negative calibration data
crossvalroc(data_mets_locs,
            data_mets_locs_EXACT,
            data_toclass_transf,
            cost,
            'SensitivityDensitySpecNeg',
            4,
            'SpecNeg')

#%% 
# plot ROC curves
# define versions
versions=['SensitivityDensityRandNeg','SensitivityDensitySpecNeg']
# loop through versions
for k in range(len(versions)):
    # define version
    version = versions[k]
    # define figure
    ax = plt.subplot(111)
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
                  (0, (2, 1)),
                  (0, (5, 1)),
                  'solid',
                  'solid',
                  (0, (2, 1)),
                  (0, (3, 1, 1, 1)),
                  'solid',
                  'solid',
                  'solid']
    
    ## calculate weighted average ROC curve
    # define empty arrays to store data
    ROC_allTP2 = np.zeros((1000,len(FSs)))
    ROC_allFP2 = np.zeros((1000,len(FSs)))
    # import data of individual curves and plot
    for b in range(len(FSs)):
        ROC_imp2 = pd.read_csv(
            '../Results/ROC_values_CrossValidation_'+FSs[b]+'_'+version+'.csv')
        ROC_allFP2[:,b] = ROC_imp2['FPrate2'].values
        ROC_allTP2[:,b] = ROC_imp2['TPrate2'].values
        plt.plot(ROC_allFP2[:,b],ROC_allTP2[:,b],label=FSs[b],
                 color=colors[b],linestyle=linestyles[b])
    # calculate weighted average ROC curve
    # define weights
    weight = [2564., 2267., 1820., 1740., 1532.,  
              962.,  543.,  524.,  492., 462.]
    # calculate average values
    TruePositive_average = np.sum(ROC_allTP2*weight,axis=1)/(np.sum(weight))
    FalsePositive_average = np.sum(ROC_allFP2*weight,axis=1)/(np.sum(weight))
    # plot average ROC curve
    plt.plot(FalsePositive_average,TruePositive_average, color='k',label='average')
    # save average ROC
    ROC_average = pd.DataFrame({'cost': cost,
                                'TruePositive_average': TruePositive_average,
                                'FalsePositive_average': FalsePositive_average})
    ROC_average.to_csv('../Results/ROC_average_'+version+'.csv',
                       index=False)
        
    # plot diagonal line
    plt.plot([0,1],[0,1],alpha=1, linewidth=0.1,color='k')
    
    # plot settings
    plt.gca().set_aspect('equal', adjustable='box')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.5 * 0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gcf().subplots_adjust(right=0.7)
    plt.title('ROC curves Cross Validation -'+version)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    #plt.savefig('../Figures/ROCs_'+version+'.png',dpi=200)
    plt.show()

#%% 
# calculate AUCs
# define function to compute AUC
def computeAUCs_average(roc):
    roc_added = pd.concat([pd.DataFrame([
            {'FalsePositive_average': 0,
            'TruePositive_average': 0}]),
            roc,
            pd.DataFrame([{'FalsePositive_average': 1,
            'TruePositive_average': 1}])])
    auc = metrics.auc(
                    roc_added.FalsePositive_average.values,
                    roc_added.TruePositive_average.values)
    return(auc)

# compute AUC ROC random calibration data
roc_density_rand = pd.read_csv('../Results/ROC_average_SensitivityDensityRandNeg.csv')
auc_density_rand = computeAUCs_average(roc_density_rand)
# export values
df_rand = pd.Series(auc_density_rand)
df_rand.to_csv('../Results/SensitivityDensityRandNeg.csv')

# compute AUC ROC negative calibration data
roc_density_neg = pd.read_csv('../Results/ROC_average_SensitivityDensitySpecNeg.csv')
auc_density_neg = computeAUCs_average(roc_density_neg)
# export values
df_neg = pd.Series(auc_density_neg)
df_neg.to_csv('../Results/SensitivityDensitySpecNeg.csv')

