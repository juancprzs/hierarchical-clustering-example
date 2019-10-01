# Data taken from
# http://techslides.com/list-of-countries-and-capitals
# and
# https://github.com/lorey/list-of-countries


import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pdb

from tqdm import tqdm
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

mapWidth    = 200
mapHeight   = 100

def mercator_proj_x(longitude):
    # Taken from 
    # https://stackoverflow.com/questions/14329691/convert-latitude-longitude-point-to-a-pixels-x-y-on-mercator-projection

    # get x value
    x = (longitude+180)*(mapWidth/360)
    return x
    

def mercator_proj_y(latitude):
    # Taken from 
    # https://stackoverflow.com/questions/14329691/convert-latitude-longitude-point-to-a-pixels-x-y-on-mercator-projection

    # convert from degrees to radians
    latRad = latitude*np.pi/180
    # get y value
    mercN = np.log(np.tan((np.pi/4)+(latRad/2)))
    y     = (mapHeight/2)-(mapWidth*mercN/(2*np.pi))
    return y

def plot_scatter(curr_clusts):
    num_sets = len(curr_clusts)
    for idx, sets in enumerate(curr_clusts.values()):
        (array, color) = sets
        for x, y in array:
            plt.scatter(x, y, c=color)


df_coords = pd.read_csv('country-capitals.csv', 
    usecols=['CountryName', 'CapitalLatitude', 'CapitalLongitude'])
df_langs = pd.read_csv('country-data.csv', sep=';',
    usecols=['area', 'capital', 'currency_name',
        'languages', 'name', 'population'])

df_langs.rename(mapper={'name' : 'CountryName'}, axis='columns', inplace=True)
df_langs = df_langs.set_index('CountryName')

df = df_coords.join(df_langs, on='CountryName')
where_valid = pd.notnull(df['languages'])
df = df.loc[where_valid]
remove_hyph_comm = lambda x: x.split(',')[0].split('-')[0]
df['languages'] = df['languages'].apply(remove_hyph_comm)

xs = df.apply(
    lambda x: mercator_proj_x(x['CapitalLongitude']),
    axis=1).values
ys = df.apply(
    lambda x: mercator_proj_y(x['CapitalLatitude']),
    axis=1).values
# Invert y values
ys = ys.max() - ys
names = df['CountryName'].values
num_items = len(ys)
color_step = 1/num_items

for idx, (x, y, n) in enumerate(zip(xs, ys, names)):
    color = cm.jet(idx / num_items)
    color = np.expand_dims(color, axis=0)
    plt.scatter(x, y, c=color)
    plt.text(x, y, n, fontsize=6)

plt.axis('off')
plt.show()

data = np.array([xs, ys]).T
clustering = linkage(data, method='centroid', metric='euclidean')

current_clusters ={
    idx : (
        np.expand_dims(x, axis=0), 
        np.expand_dims(np.array(cm.jet(idx/num_items)), axis=0)
    ) for idx, x in enumerate(data)
}

if False:
    plt.figure()
    plt.ion()
    for idx, clust in tqdm(enumerate(clustering)):
        merged_1 = int(clust[0])
        (merged_1, c1) = current_clusters.pop(merged_1)
        merged_2 = int(clust[1])
        (merged_2, c2) = current_clusters.pop(merged_2)
        key = idx + num_items
        new_c = (c1 + c2) / 2
        new_array = np.concatenate((merged_1, merged_2), axis=0)
        current_clusters[key] = (new_array, new_c)
        if idx % 4 == 0:
            plt.clf()
            plot_scatter(current_clusters)
            plt.title(f'Iteration {idx} - clusters: {len(current_clusters)}')
            plt.axis('off')
            plt.draw()
            plt.pause(.2)


plt.figure()
dendrogram = sch.dendrogram(clustering, orientation='right', labels=names)
plt.show()
