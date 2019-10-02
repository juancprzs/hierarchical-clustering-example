# Data taken from
# http://techslides.com/list-of-countries-and-capitals
# and
# https://github.com/lorey/list-of-countries


import os
import copy
import imageio
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

import pdb

from glob import glob
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
    for sets in curr_clusts.values():
        (array, color) = sets
        for x, y in array:
            plt.scatter(x, y, c=color)


def plot_iterations(dir_name, clustering, current_clusters):
    plt.figure(
        num=None, figsize=(14, 9), dpi=140, facecolor='w', edgecolor='k')
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False,
    )
    for idx, clust in tqdm(enumerate(clustering)):
        merged_1 = int(clust[0])
        (merged_1, c1) = current_clusters.pop(merged_1)
        merged_2 = int(clust[1])
        (merged_2, c2) = current_clusters.pop(merged_2)
        key = idx + num_items
        new_c = (c1 + c2) / 2
        new_array = np.concatenate((merged_1, merged_2), axis=0)
        current_clusters[key] = (new_array, new_c)
        if idx % 1 == 0:
            plt.clf()
            plot_scatter(current_clusters)
            dist_fun = dir_name.split('_')[0]
            title = f'{dist_fun} - Iteration {idx} - clusters: {len(current_clusters)}'
            plt.title(title, fontsize=18)
            plt.grid(True)
            plt.tick_params(
                axis='both',       # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False,
                left=False,
                labelleft=False,
            )
            plt.tight_layout()
            plt.draw()
            figname = osp.join(dir_name, f'iteration_{idx}.png')
            plt.savefig(figname)

    plt.close()


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

df['x-coord'] = df.apply(
    lambda x: mercator_proj_x(x['CapitalLongitude']),
    axis=1)
df['y-coord'] = df.apply(
    lambda x: mercator_proj_y(x['CapitalLatitude']),
    axis=1)
# Invert y values
ys = df['y-coord'].values
df['y-coord'] = ys.max() - ys
# Save the dataframe
df.to_csv('final_countries_data.csv', sep=',', index=False)

xs = df['x-coord'].values
ys = df['y-coord'].values
names = df['CountryName'].values
num_items = len(ys)
color_step = 1/num_items
data = np.array([xs, ys]).T

plt.figure(
    num=None, figsize=(14, 9), dpi=140, facecolor='w', edgecolor='k')
for idx, (x, y, n) in enumerate(zip(xs, ys, names)):
    color = cm.jet(idx / num_items)
    color = np.expand_dims(color, axis=0)
    plt.scatter(x, y, c=color)
    plt.text(x, y, n, fontsize=12)

plt.grid(True)
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False,
)
plt.title('Some countries of the world', fontsize=18)
plt.tight_layout()
plt.savefig('world_map_mercator.png')
plt.close()

current_clusters = {
    idx : (
        np.expand_dims(x, axis=0), 
        np.expand_dims(np.array(cm.jet(idx/num_items)), axis=0)
    ) for idx, x in enumerate(data)
}

methods = ['single', 'complete', 'average', 'centroid', 'ward']
for method in methods:
    dir_name = f'{method}_euclidean'
    clustering = linkage(data, method=method, metric='euclidean')
    if not osp.exists(dir_name):
        print(f'Saving on {dir_name}')
        os.makedirs(dir_name)
        curr_clusts = copy.deepcopy(current_clusters)
        plot_iterations(dir_name=dir_name, 
            clustering=clustering, current_clusters=curr_clusts)
    else:
        print('Figures already existed')
        plt.figure(num=None, figsize=(14, 9), dpi=140, facecolor='w', edgecolor='k')
        dendrogram = sch.dendrogram(clustering, orientation='right', labels=names)
        figname = osp.join(dir_name, 'dendrogram.png')
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()



for method in methods:
    dir_name = f'{method}_euclidean'
    figname = osp.join(dir_name, 'clustering_animation.gif')
    if osp.exists(figname):
        print('GIF already existed')
        continue
    print(f'Generating GIF for {dir_name}')
    filenames = sorted(glob(
        osp.join(dir_name, 'iteration_*.png')),
        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    imageio.mimsave(figname, images)
