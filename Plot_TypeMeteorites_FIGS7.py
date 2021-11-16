# import packages
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.colors

# set working directory
path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

#%%
# import meteorite table
mets = pd.read_csv('../Data_raw/meteorite_locations_raw.csv', low_memory=False)
# group data by Type
types = mets.groupby('Type',as_index=False).count()


# clean data (manually)
types['Type'] = types['Type'].replace(['Acapulcoite/lodranite'],'Acapulcoite/Lodranite')

types['Type'] = types['Type'].replace(['Chondite-ung'],'C')
types['Type'] = types['Type'].replace(['CBb'],'CB')
types['Type'] = types['Type'].replace(['Chondrite-ung'],'C')
types['Type'] = types['Type'].replace(['Chondrite-uncl'],'C')
types['Type'] = types['Type'].replace(['Chondrite-fusion crust'],'C')

types['Type'] = types['Type'].replace(['Stone-ung'],'Rest')
types['Type'] = types['Type'].replace(['Stone-uncl'],'Rest')
types['Type'] = types['Type'].replace(['Terrestrial rock'],'Rest')
types['Type'] = types['Type'].replace(['Unknown'],'Rest')
types['Type'] = types['Type'].replace(['Pseudometeorite'],'Rest')
types['Type'] = types['Type'].replace(['Doubtful meteorite'],'Rest')
types['Type'] = types['Type'].replace(['Fusion crust'],'Rest')

types['Type'] = types['Type'].replace(['Ureilite-pmict'],'Ureilite')

types['Type'] = types['Type'].replace(['Diogenite-olivine'],'Diogenite')
types['Type'] = types['Type'].replace(['Diogenite-pm'],'Diogenite')

types['Type'] = types['Type'].replace(['Pallasite, PMG'],'Pallasite')
types['Type'] = types['Type'].replace(['Pallasite, ungrouped'],'Pallasite')

types['Type'] = types['Type'].replace(['Mesosiderite-A'],'Mesosiderite')
types['Type'] = types['Type'].replace(['Mesosiderite-B'],'Mesosiderite')
types['Type'] = types['Type'].replace(['Mesosiderite-B1'],'Mesosiderite')

types['Type'] = types['Type'].replace(['Martian (OPX)'],'Orthopyroxenite')
types['Type'] = types['Type'].replace(['Martian (nakhlite)'],'Nakhlite')
types['Type'] = types['Type'].replace(['Martian (shergottite)'],'Shergottite')

types['Type'] = types['Type'].replace(['Lunar (anorth)'],'Lunar')
types['Type'] = types['Type'].replace(['Lunar (bas/anor)'],'Lunar')
types['Type'] = types['Type'].replace(['Lunar (basalt)'],'Lunar')
types['Type'] = types['Type'].replace(['Lunar (feldsp. breccia)'],'Lunar')
types['Type'] = types['Type'].replace(['Lunar (gabbro)'],'Lunar')


for p in range(len(types)):
    if types['Type'].iloc[p][-3::] == '-an':
        types['Type'].iloc[p] = types['Type'].iloc[p][0:-3]
    if types['Type'].iloc[p][-4::] == '-ung':
        #print(types['Type'].iloc[p])
        types['Type'].iloc[p] = types['Type'].iloc[p][0:-4]
        #print(types['Type'].iloc[p])
    if types['Type'].iloc[p][-10::] == '-melt rock':
        #print(types['Type'].iloc[p])
        types['Type'].iloc[p] = types['Type'].iloc[p][0:-10]
        #print(types['Type'].iloc[p])
    if types['Type'].iloc[p][:4] == 'Iron':
        #print(types['Type'].iloc[p])
        if len(types['Type'].iloc[p])>4:
            if len(types['Type'].iloc[p])>9:
                #print('longer than 9')
                if (types['Type'].iloc[p][9]=='-'):
                    types['Type'].iloc[p] = types['Type'].iloc[p][:9]
            types['Type'].iloc[p] = types['Type'].iloc[p][6:]
            
        #print(types['Type'].iloc[p])
    if types['Type'].iloc[p][:7] == 'Eucrite':
        #print(types['Type'].iloc[p])
        types['Type'].iloc[p] = types['Type'].iloc[p][:7]
        #print(types['Type'].iloc[p])
    if types['Type'].iloc[p][-13::] == '-melt breccia':
        #print(types['Type'].iloc[p])
        types['Type'].iloc[p] = types['Type'].iloc[p][0:-13]
        #print(types['Type'].iloc[p])
    if types['Type'].iloc[p][-6::] == '-metal':
        #print(types['Type'].iloc[p])
        types['Type'].iloc[p] = types['Type'].iloc[p][0:-6]
        #print(types['Type'].iloc[p])
    if types['Type'].iloc[p][-9::] == '-imp melt':
        #print(types['Type'].iloc[p])
        types['Type'].iloc[p] = types['Type'].iloc[p][0:-9]
        #print(types['Type'].iloc[p])
        
types['Type'] = types['Type'].replace(['ungrouped'],'Iron')
types['Type'] = types['Type'].replace(['ungrouped¶'],'Iron')
types['Type'] = types['Type'].replace(['IAB complex'],'IAB')
types['Type'] = types['Type'].replace(['IIE?'],'IIE')
# types['Type'] = types['Type'].replace(['Lunar (gabbro)'],'Lunar')

# convert strings to get rid of spaces/other punctuation marks and numbers
types['Str_conv'] = types['Type'].replace('(\W)', '', regex=True)
types['Str_conv2'] = types['Str_conv'].replace('(\d)','', regex=True)
# define empty array to mark whether a row in the dataframe has been accounted for or not
types['counted'] = np.zeros(len(types))


#%%
# create groups of meteorite classes
# groupings inspired by https://curator.jsc.nasa.gov/antmet/statistics.cfm and
# http://www.meteoritemarket.com/TypeDiagram1.pdf and https://en.wikipedia.org/wiki/Angrite


# define empty lists to fill with super classes and clases
supclas = []
clas = []

# define groups within the super classes
groups_chond =  ['C','CI','CM','CO','CV','CK','CR','CH','CB',
        'H','L','LL','LLL','HL','OC','E','EH','EL','R','K']
#C = C ungrouped
#E = E ungrouped
#LLL = L/LL
#HL = H/L
#OC = O ungrouped
groups_achond = ['Angrite','Aubrite','Eucrite','Diogenite','Howardite','Mesosiderite',
               'Pallasite',
               'Lunar','Shergottite','Nakhlite','Orthopyroxenite','Achondrite']#'Chassignite',
groups_primachond = ['Ureilite','Brachinite','Acapulcoite',
                     'AcapulcoiteLodranite','Lodranite',
                     'Winonaite']
groups_irons = ['IAB','IIAB','IID','IIE','IIIAB','IVA','IVB','Iron']
#Achondrite = Achondrite ungrouped
groups_rest = ['Rest']

# define empty array to count number of meteorites per group
counts = np.zeros(len(groups_chond)+len(groups_achond)+len(groups_primachond)+len(groups_irons)+1)

# count number of meteorites per group and assign values to super classes and classes
for i in range(len(groups_chond)):
    supclas.append('Chondrites')
    if i < 9:
        clas.append('Carbonaceous')
    if 9 <= i < 15:
        clas.append('Ordinary')
    if 15 <= i < 18:
        clas.append('Enstatite')
    if i>=18:
        clas.append(groups_chond[i])
    #print(groups_chond[i]+'---------------')
    #print(types[types['Str_conv2'].str.contains(groups_chrond[i])][['Str_conv2','Name']])
    #print(types[types['Str_conv2']==groups_chond[i]][['Type','Name']])
    types['counted'].iloc[types['Str_conv2']==groups_chond[i]] = 1.0
    counts[i] = sum(types[types['Str_conv2']==groups_chond[i]]['Name'])

for j in range(len(groups_achond)):
    supclas.append('Achondrites')
    clas.append('ni')
#     print(groups_achond[j]+'---------------')
#     print(types[types['Str_conv2']==groups_achond[j]][['Type','Name']])
    types['counted'].iloc[types['Str_conv2']==groups_achond[j]] = 1.0
    counts[len(groups_chond)+j] = sum(types[types['Str_conv2']==groups_achond[j]]['Name'])

for k in range(len(groups_primachond)):
    supclas.append('Primitive Achondrites')
    clas.append('ni')
#     print(groups_primachond[k]+'---------------')
#     print(types[types['Str_conv2']==groups_primachond[k]][['Type','Name']])
    types['counted'].iloc[types['Str_conv2']==groups_primachond[k]] = 1.0
    counts[len(groups_chond)+len(groups_achond)+k] = sum(types[types['Str_conv2']==groups_primachond[k]]['Name'])
    
for n in range(len(groups_irons)):
    supclas.append('Irons')
    clas.append('ni')
    types['counted'].iloc[types['Str_conv2']==groups_irons[n]] = 1.0
    counts[len(groups_chond)+len(groups_achond)+len(groups_primachond)+ n] = sum(types[types['Str_conv2']==groups_irons[n]]['Name'])

supclas.append('Unkown')
clas.append('ni')
#     print(groups_primachond[k]+'---------------')
#     print(types[types['Str_conv2']==groups_primachond[k]][['Type','Name']])
types['counted'].iloc[types['Str_conv2']=='Rest'] = 1.0
counts[len(groups_chond)+len(groups_achond)+len(groups_primachond)+len(groups_irons)] = sum(
    types[types['Str_conv2']=='Rest']['Name'])

# check whether all types are accounted for
print(sum(types.counted))

# check which rows are not accounted for in the grouping of meteorites per group/class
types[types.counted==0]

#%%
# plot data
# reorganize the data in a dataframe
mets_class_org = pd.DataFrame({'Supclas': supclas, 
                               'Class':clas,
                               'Group':groups_chond+groups_achond+groups_primachond+groups_irons+groups_rest,
                               'count':counts})

# check whether all meteorites have been counted only once
print(sum(mets_class_org['count']),len(mets))

# clearify labels data
mets_class_org['Group'] = mets_class_org['Group'].replace(['C'],'C ungrouped')
mets_class_org['Group'] = mets_class_org['Group'].replace(['E'],'E ungrouped')
mets_class_org['Group'] = mets_class_org['Group'].replace(['LLL'],'L/LL')
mets_class_org['Group'] = mets_class_org['Group'].replace(['HL'],'H/L')
mets_class_org['Group'] = mets_class_org['Group'].replace(['OC'],'O ungrouped')
mets_class_org['Group'] = mets_class_org['Group'].replace(['Iron'],'Iron ungrouped')
mets_class_org['Group'] = mets_class_org['Group'].replace(['Achondrite'],'Achondrite ungrouped')
mets_class_org['Group'] = mets_class_org['Group'].replace(['AcapulcoiteLodranite'],'Acapulcoite/Lodranite')
mets_class_org['Group'] = mets_class_org['Group'].replace(['Rest'],'Unkown/Terrestrial')
mets_class_org['Class'] = mets_class_org['Class'].replace(['ni'],' ')
#mets_class_org['Class'] = mets_class_org['Class'].replace(['ni'],' ')

# assign the labels in the dataframe
pltlabels = []
for m in range(len(mets_class_org)):
    pltlabels.append(mets_class_org['Group'][m]+' ('+str(int(mets_class_org['count'][m]))+')')




#%%
## plot figure
# define figure
fig, axs = plt.subplots(1,1, figsize=((18./2.54,30./2.54)))

## define colors (https://stackoverflow.com/questions/57720935/how-to-use-correct-cmap-colors-in-nested-pie-chart-in-matplotlib)
# define function to create colormaps
def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap
# define number of classes, number of subclasses and a colormap
n_classes = 5
n_subclasses = len(groups_chond)
cmap = categorical_cmap(n_classes,n_subclasses)
# create array of colors for piechart
colors_1 = cmap(np.arange(0,n_classes)*len(groups_chond))
# create array of colors for barplot (hardcoded)
ar3 = np.concatenate((np.linspace(1,19,len(groups_chond)).astype(int), 
                     np.linspace(21,39,len(groups_achond)).astype(int),
                     np.linspace(41,59,len(groups_primachond)).astype(int),
                     np.linspace(61,79,len(groups_irons)).astype(int),
                     np.linspace(81,99,len(groups_rest)).astype(int)))
colors_3 = cmap(ar3)


## plot barplot
# define length of bars
ratios = mets_class_org['count'].values
# define corresponding labels of bars
labels = mets_class_org['Group'].values
# plot bars
plt.barh(labels, ratios, height=0.8,color=colors_3,zorder=0)
# set xlimit
plt.xlim([0,170])

# create arrowlike end of bars larger than xlimit
for k in range(len(ratios)):
    if ratios[k]>170:
        plt.scatter(169.5,k+.42,marker='D',s=121,color='white',zorder=1,alpha=1)
        plt.scatter(169.5,k-.5,marker='D',s=121,color='white',zorder=1,alpha=1)

# plot values corresponding to length of bars
# transform values into strings
values_ratios = ratios.astype(int).astype(str)
# loop through all bars
for i in range(len(values_ratios)):
    # set coordinates of plotting location
    x_value = ratios[i]
    y_value = float(i)
    # for bars within xlimit
    if ratios[i]<170:
        plt.annotate(
                values_ratios[i], # label to plot
                (10,1), # point to annotate (arbitrary)
                xytext=(x_value+1, y_value), # location of the label
                ha='left', # horizontal alignment
                va='center', # vertical alignment
                fontsize=8.5)
    # for bars larger than xlimit
    if ratios[i]>=170:
        plt.annotate(
                values_ratios[i], # label to plot
                (10,1), # point to annotate (arbitrary)
                xytext=(165, y_value), # location of the label
                ha='right', # horizontal alignment
                va='center', # vertical alignment
                fontsize=8.5)

# plot settings
plt.ylim([-1,len(values_ratios)])        
plt.gca().invert_yaxis()
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)

# plot labels
plt.xlabel('counts')
plt.ylabel('meteorite group',labelpad=-20)
plt.title('Meteorite Types (n='+str(int(sum(mets_class_org['count'])))+')')

## plot pie diagram
axs2=axs.twiny()

# group data by supclas
met_class_org_gr = mets_class_org.groupby('Supclas', sort=False).sum()

# plot pie diagram
centerp,labels_ = axs2.pie(met_class_org_gr['count'], 
          radius=2.8,
          center=(1.5,40),
          colors=colors_1,
          wedgeprops=dict(linewidth=0., edgecolor='w'),
          rotatelabels=False,
          startangle=-40,
          frame=True);
# set title of pie diagram (manually)
axs2.annotate('meteorite type',
                (7,40), # arbitrary point
                xytext=(1.5, 36.5), # location of the label (wrt arbitray point)
                ha='center', # horizontal alignment
                va='center', # vertical alignment
                fontsize=10)
# plot settings
axs2.set_xticks([])
axs2.spines['right'].set_visible(False)
axs2.spines['top'].set_visible(False)
# plot legend
plt.legend(centerp,met_class_org_gr.index.values,loc='lower right', bbox_to_anchor=(1.12, 0.092));
# adjust spacing and save figure
plt.subplots_adjust(left=0.23, bottom=0.04, right=0.92, top=0.98, wspace=0)
#plt.savefig('../Figures/MetTypes.png',dpi=300,facecolor='white')
