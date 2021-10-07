# import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter 
from matplotlib import colors
import os
import pandas as pd
from matplotlib import ticker
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import itertools

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# run exhaustive feature selection (only combinations with Temperature and Velocity)
# import function from "ExhaustiveFeatureSelection_CombinationsWithTemperatureVelocity.py"
from ExhaustiveFeatureSelection_CombinationsWithTemperatureVelocity import exhaustive_feature_selection

# define combinations of definitions of surface slope and surface temperature
# combinations_slope_stemp = [['slope5km','stempPERC70'],
#                             ['slope2km','stempPERC70'],
#                             ['slope400m','stempPERC70'],
#                             ['slope5km','stempPERC90'],
#                             ['slope2km','stempPERC90'],
#                             ['slope400m','stempPERC90'],
#                             ['slope5km','stempPERC95'],
#                             ['slope2km','stempPERC95'],
#                             ['slope400m','stempPERC95'],
#                             ['slope5km','stempPERC99'],
#                             ['slope2km','stempPERC99'],
#                             ['slope400m','stempPERC99']]
combinations_slope_stemp =   [['slope2km','stempPERC99']]

# loop through combinations of surface slope and surface temperature, with random validation data                    
for i in range(len(combinations_slope_stemp)):
    print('STARTED:',combinations_slope_stemp[i])
    exhaustive_feature_selection(combinations_slope_stemp[i],
                                  'RandNeg')

# loop through combinations of surface slope and surface temperature, with negative validation data
for i in range(len(combinations_slope_stemp)):
    print('STARTED:',combinations_slope_stemp[i])
    exhaustive_feature_selection(combinations_slope_stemp[i],
                                  'SpecNeg')

#%%
# run exhaustive feature selection (all other combinations, without Temperature and Velocity)
# import function from "ExhaustiveFeatureSelection_AllOtherCombinations.py"
from ExhaustiveFeatureSelection_AllOtherCombinations import exhaustive_feature_selection

# define combinations of definitions of surface slope and surface temperature
combinations_slope_stemp = [['slope2km','stempPERC99']]

# loop through combinations of surface slope and surface temperature, with random validation data                    
for i in range(len(combinations_slope_stemp)):
    print('STARTED:',combinations_slope_stemp[i])
    exhaustive_feature_selection(combinations_slope_stemp[i],
                                  'RandNeg')

# loop through combinations of surface slope and surface temperature, with negative validation data  
for i in range(len(combinations_slope_stemp)):
    print('STARTED:',combinations_slope_stemp[i])
    exhaustive_feature_selection(combinations_slope_stemp[i],
                                  'SpecNeg')