# import packages
import numpy as np
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

import geopandas
from shapely.geometry import Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# set front properties
fontprops = fm.FontProperties(size=10)

#%%
# read in data
MSZs_unsorted = geopandas.read_file('../Data_Locations/MSZs_ranked.shx')
# sort data
MSZs = MSZs_unsorted.sort_values(by=['rank_total']).reset_index()
#%%
# function to open basemap given lower left coordinates
def openJPGgivenbounds(xmin,ymin):
    # define arrays referring to names of LIMA subtiles
    cirref_xs = np.arange(-2700000, 2700000, 150000)
    cirref_ys = np.arange(-2700000, 2700000, 150000)
    # select names of LIMA subtiles corresponding to given coordinates
    cirref_x = cirref_xs[cirref_xs<xmin][-1]
    cirref_y = cirref_ys[cirref_ys<ymin][-1]
    # define list of zeros to make sure names include zeros
    name_zeros = list('0000000')
    # ensure a minus is included in namestring when given coordinates are negative
    if cirref_x < 0:
        name_zeros[7-len(str(abs(cirref_x))):] = list(str(abs(cirref_x)))
        name_x_abs = ''.join(list(name_zeros))
        name_x = '-'+name_x_abs
        name_zeros = list('0000000')
    else:
        name_zeros[7-len(str(abs(cirref_x))):] = list(str(abs(cirref_x)))
        name_x_abs = ''.join(list(name_zeros))
        name_x = '+'+name_x_abs
        name_zeros = list('0000000')
    if cirref_y < 0:
        name_zeros[7-len(str(abs(cirref_y))):] = list(str(abs(cirref_y)))
        name_y_abs = ''.join(list(name_zeros))
        name_y = '-'+name_y_abs
        name_zeros = list('0000000')
    else:
        name_zeros[7-len(str(abs(cirref_y))):] = list(str(abs(cirref_y)))
        name_y_abs = ''.join(list(name_zeros))
        name_y = '+'+name_y_abs
        name_zeros = list('0000000')
    # define full name of LIMA subtile
    name = 'CIRREF_'+'x'+name_x+'y'+name_y
    # try open LIMA subtile (for high latitudes no data exists)
    try:
        img = rasterio.open('../Data_raw/LIMA/'+name+'.jpg')
    except:
        print('no high resolution background image')
        img = 0
    # return image and lowerleft coordinates of image
    return(img,cirref_x,cirref_y)
#%%
# define function that creates plot of a given meteorite stranding zone
def plot_MSZ_table(n_geometry, # number of the geometry
                   loc_scalebar = 'lower right', # location of the scalebar
                   margin_additional_jpg=0.05, # margin adjustment needed if background images are missing in the plot
                   colorlabels='black' # set the color of the labels
                   ):
    # open basemap
    # calculate bounds
    xmin,ymin,xmax,ymax = MSZs.loc[[n_geometry],'geometry'].bounds.values[0,:]
    # open jpg
    img,cirref_x,cirref_y = openJPGgivenbounds(xmin,ymin)
    # define figure
    fig, ax1 = plt.subplots(1,1)
    # define figure size
    figwidth = 3.4 # in cm
    fig.set_figwidth(figwidth/2.54)
    # plot background image (if available) and save whether background is available
    try:
        show(img.read(), ax=ax1, transform=img.transform)
        background = 1
    except:
        print('')
        background = 0
        
    # plot additional background images if needed given bounds of polygon and defined margin
    if 1*xmax + margin_additional_jpg*abs(xmax) > cirref_x + 150000:
        print('additional jpg needed')
        img_add_x,cirref_x,cirref_y = openJPGgivenbounds(xmax,ymin)
        try:
            show(img_add_x.read(), ax=ax1, transform=img_add_x.transform)
        except:
            print('no additional jpg available')
    if 1*ymax + margin_additional_jpg*abs(ymax) > cirref_y + 150000:
        print('additional jpg needed')
        img_add_y,cirref_x,cirref_y = openJPGgivenbounds(xmax,ymax)
        try:
            show(img_add_y.read(), ax=ax1, transform=img_add_y.transform)
        except:
            print('no additional jpg available')
    if 1*xmin - margin_additional_jpg*abs(xmin) < cirref_x:
        print('additional jpg needed')
        img_add_x_left,cirref_x,cirref_y = openJPGgivenbounds(1*xmin - margin_additional_jpg*abs(xmin),ymin)
        try:
            show(img_add_x_left.read(), ax=ax1, transform=img_add_x_left.transform)
        except:
            print('no additional jpg available')
    if 1*ymin - margin_additional_jpg*abs(ymin) < cirref_y:
        print('additional jpg needed')
        img_add_y_lower,cirref_x,cirref_y = openJPGgivenbounds(xmin,1*ymin - margin_additional_jpg*abs(ymin))
        try:
            show(img_add_y_lower.read(), ax=ax1, transform=img_add_y_lower.transform)
        except:
            print('no additional jpg available')
    
    # set limits of plot
    # calculate the span of the polygon
    x_dist = xmax - xmin
    y_dist = ymax - ymin
    # define ratio of image
    max_ratio = 0.75
    if y_dist/x_dist > max_ratio: # too high
        # save whether original polygon is too high or too wide
        toohigh = 1
        # rescale defined ratio
        max_ratio_div = 1/max_ratio
        # calculate addition to xlimit 
        x_lim_add = (1.4*max_ratio_div*y_dist - x_dist)/2
        # set xlimit
        plt.xlim([xmin-x_lim_add,xmax+x_lim_add])
        # recalculate x_dist
        x_dist = y_dist*max_ratio_div
        # set ylimit
        plt.ylim([ymin-0.2*y_dist,ymax+0.2*y_dist])# ylim normal
        
    else: # too wide
        # save whether original polygon is too high or too wide
        toohigh = 0
        # calculate addition to ylimit
        y_lim_add = (1.4*max_ratio*x_dist - y_dist)/2
        # set ylimit
        plt.ylim([ymin-y_lim_add,ymax+y_lim_add])
        # recalculate y_dist
        y_dist = x_dist*max_ratio
        # set xlimit
        plt.xlim([xmin-0.2*x_dist,xmax+0.2*x_dist]) # xlim normal
        

    # set figure height
    figheight = (figwidth/2.54)*(1.2*y_dist)/(1.2*x_dist)
    fig.set_figheight(figheight)
    
    # add scalebar
    scalebar = AnchoredSizeBar(ax1.transData,
                               2000, '2 km', loc_scalebar, 
                               pad=0.01, #0.0005
                               color=colorlabels,
                               frameon=False, # size_vertical=10000*300/x_dist,
                               fontproperties=fm.FontProperties(size=7),
                               label_top=True,
                               sep=1)
    ax1.add_artist(scalebar)

    # add northarrow
    # set arrowlength
    arrow_length = 0.05*y_dist
    # calculate center of plot
    x_center = (xmax+xmin)/2
    y_center = (ymax+ymin)/2
    # calculate additional length of arraw
    x_add = arrow_length*x_center/((x_center**2 + y_center**2)**0.5)
    y_add = arrow_length*y_center/((x_center**2 + y_center**2)**0.5)
    # calculate location of arrow
    if toohigh == 1:
        x_arrow = xmax+x_lim_add - 0.1*x_dist
        y_arrow = ymax +0.1*y_dist
    else:
        x_arrow = xmax +0.1*x_dist
        y_arrow = ymax+y_lim_add - 0.1*y_dist
    # plot arrow
    plt.arrow(x_arrow-x_add,y_arrow-y_add,2*x_add,2*y_add,color=colorlabels,
              width=0.005*x_dist,length_includes_head=True) #width=40
    
    # plot MSZ
    # plot all MSZs
    MSZs['geometry'].boundary.plot(ax=ax1,edgecolor='black',linewidth=0.3)
    # plot specific MSZ
    MSZs.loc[[n_geometry],'geometry'].boundary.plot(ax=ax1,edgecolor='black',linewidth=1.2)
    
    # plot settings
    plt.axis('off')
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    # save figure
    fig.savefig('../Figures/MSZs_vis3/MSZ_vis'+str(n_geometry+1)+'.png',dpi=300)
    return(background)

#%%
# define empty array to save whether a background is available
background = np.zeros(len(MSZs))
# loop through range of meteorite stranding zones to produce figures
for i in range(0,10):
    background[i] = plot_MSZ_table(i)
    print(i)
# manually adjust certain plots (optional)
# plot_MSZ_table(8,margin_additional_jpg=0.1) 
# plot_MSZ_table(12,margin_additional_jpg=0.1)   
#%%
# reorganize data and generate table 
# -*- coding: utf-8 -*-
#MSZs['background_'] = background
MSZs['rank'] = (MSZs.index+1).astype(str) + ' (' + MSZs.rank_total.astype(int).astype(str) + ')' 
lat = abs(MSZs['ycoord']).round(3).astype(str)
lon = MSZs['xcoord'].round(3).astype(str)
MSZs['location'] = lat + '°S, ' + lon + '°E'
MSZs['temperature'] = MSZs['maxtemp_me'].round(1).astype(str)
MSZs['velocity'] = MSZs['speed_medi'].round(1).astype(str)
MSZs['days_snowfree'] = (np.round(MSZs['3snow_free']/200,0)*10).astype(int).astype(str) # 2000-02-18T00:00:00 - 2021-02-09T00:00:00
MSZs['area_'] = MSZs['area'].round(1).astype(str)
DNS_rounded = (np.round(MSZs['1distance_']/10,0)*10).astype(int).astype(str)
MSZs['nearest_station'] = MSZs['nearest_st'] + ' (' + DNS_rounded + ')'
MSZs_tosave = MSZs[['rank','location','temperature','velocity',
                   'days_snowfree','area_','nearest_station']]
MSZs_tosave.to_csv('../Results/613MSZs_sorted.csv',encoding='utf-8-sig')