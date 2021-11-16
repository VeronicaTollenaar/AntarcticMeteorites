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

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# run cross validation with defined number of principal components
# import function from "VariousPCs.py"
from VariousPCs import number_pcs

# define combination of definitions of surface slope and surface temperature
combinations_slope_stemp =   [['slope2km','stempPERC99']]
                           
# loop through different numbers of principal components
for i in range(2,4):
    number_pcs(
        combinations_slope_stemp[0], # distance over which slope is calculated, percentile of surface temperature
        'RandNeg', # type of negative validation data (specific (negative) or random)
        (0,1,3,4), # selected features (0=radar, 1=velocity, 2=icethickness, 3=slope, 4=temperature, 5=distance to outcrops)
        i # number of principal components considered
        )
