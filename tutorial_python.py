import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

T = pd.read_csv('final_countries_data.csv')
coords = T[['x-coord', 'y-coord']].values
names = T['CountryName'].values
plt.figure()
plt.subplot(211)
for (xx, yy), name in zip(coords, names):
    plt.scatter(xx, yy)
    plt.text(xx, yy, name, fontsize=6)

plt.title('Countries')
plt.grid(True)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticks([])

colors = ['r', 'g', 'b', 'm']
D = pdist(coords, 'euclidean')
Z = linkage(D, method='single')
clst = fcluster(Z, t=4, criterion='maxclust')

plt.subplot(212)
for color, tclust in zip(colors, np.unique(clst)):
    where = tclust==clst
    plt.scatter(coords[where, 0], coords[where, 1], c=color)

plt.title('Clusters')
plt.grid(True)
plt.gca().get_yaxis().set_ticks([])
plt.gca().get_xaxis().set_ticks([])
plt.tight_layout()
plt.show()