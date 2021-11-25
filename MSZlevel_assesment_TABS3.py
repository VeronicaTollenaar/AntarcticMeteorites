import numpy as np
import os
import geopandas

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
#%%
# reorganize negative observations for testing
negative_obs_test = geopandas.read_file('../Data_Locations/Test_neg4326.shx')
negative_obs_test = negative_obs_test.sort_values(by=['Name'],key=lambda col: col.str.lower()).reset_index()
lat = abs(negative_obs_test['ycoord']).round(3).astype(str)
lon = negative_obs_test['xcoord'].round(3).astype(str)
negative_obs_test['location'] = lat + '°S, ' + lon + '°E'
negative_obs_test['temperature'] = negative_obs_test['_tempmean'].round(1).astype(str) + ' °C'
negative_obs_test['velocity'] = negative_obs_test['_speedmean'].round(1).astype(str) + ' m/yr'
negative_obs_test['slope'] = np.round(negative_obs_test['_slopemean']*1000,1).astype(str) + ' m/km'
negative_obs_test['radar'] = np.round(negative_obs_test['_radarmean'],0).astype(int).astype(str) + ' -'
negative_obs_test['n_obs'] = np.round(negative_obs_test['_tempcount'],0).astype(int).astype(str)
negative_obs_test['perc_class'] = np.round(negative_obs_test['_perc_pos_']*100,0).astype(int).astype(str) + ' %'

negative_obs_test_tosave = negative_obs_test[['Name','n_obs','perc_class','location','temperature','velocity',
                    'radar','slope']]
negative_obs_test_tosave.to_csv('../Results/negative_obs_test_org.csv',encoding='utf-8-sig')

#%%
# reorganize negative observations for calibration
negative_obs_cal = geopandas.read_file('../Data_Locations/Cal_neg4326.shx')
negative_obs_cal = negative_obs_cal.sort_values(by=['Name'],key=lambda col: col.str.lower()).reset_index()

lat = abs(negative_obs_cal['ycoord']).round(3).astype(str)
lon = negative_obs_cal['xcoord'].round(3).astype(str)
negative_obs_cal['location'] = lat + '°S, ' + lon + '°E'

negative_obs_cal_tosave = negative_obs_cal[['Name','location']]
negative_obs_cal_tosave.to_csv('../Results/negative_obs_cal_org.csv',encoding='utf-8-sig')
#%%
# reorganize positive observations for testing on MSZ level
positive_obs_test = geopandas.read_file('../Data_Locations/TestMSZs_pos4326.shx')
positive_obs_test = positive_obs_test.sort_values(by=['Name'],key=lambda col: col.str.lower()).reset_index()

lat = abs(positive_obs_test['ycoord']).round(3).astype(str)
lon = positive_obs_test['xcoord'].round(3).astype(str)
positive_obs_test['location'] = lat + '°S, ' + lon + '°E'
positive_obs_test['temperature_'] = positive_obs_test['temperatur'].round(1).astype(str) + ' °C'
positive_obs_test['velocity_'] = positive_obs_test['velocity'].round(1).astype(str) + ' m/yr'
positive_obs_test['slope_'] = np.round(positive_obs_test['slope']*1000,1).astype(str) + ' m/km'
positive_obs_test['radar_'] = np.round(positive_obs_test['radar'],0).astype(int).astype(str) + ' -'
positive_obs_test['perc_class_'] = np.round(positive_obs_test['perc_class']*100,0).astype(int).astype(str) + ' %'

positive_obs_test_tosave = positive_obs_test[['Name','location','temperature_','velocity_',
                    'radar_','slope_','perc_class_']]
positive_obs_test_tosave.to_csv('../Results/positive_obs_test_org.csv',encoding='utf-8-sig')

