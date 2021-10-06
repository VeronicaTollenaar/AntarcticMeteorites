# import packages
import numpy as np
import os
import pandas as pd

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# import final classification values
version = 'slope2kmstempPERC99_4pcs4fs0134'
classified_pos = pd.read_csv('../Results/positive_classified_'+version+'.csv')
classified_neg = pd.read_csv('../Results/negative_classified_'+version+'.csv')
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
                 axis=1)[:]

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

#%%
TP = len(classified_pos.merge(test_pos))
print(TP)
FN = len(classified_neg.merge(test_pos))
print(FN)

#%%
FP = len(classified_pos.merge(test_neg))
print(FP)
TN = len(classified_neg.merge(test_neg))
print(TN)

#%%
ACC = (TP + TN) /(TP + TN + FP + FN)
REC = TP /(TP + FN)
SPEC = TN /(TN + FP)
PREC = TP / (TP + FP)
F1 = 2*TP/(2*TP + FP + FN)
print('accuracy:', np.round(ACC,3)*100,
      '\nsensitivity:', np.round(REC,3)*100,
      '\nspecificity:', np.round(SPEC,3)*100,
      '\nprecision:',np.round(PREC,3)*100,
      '\nF1:',np.round(F1,3)*100)
