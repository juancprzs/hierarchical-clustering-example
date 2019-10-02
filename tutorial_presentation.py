import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import (pdist, squareform)
from scipy.cluster.hierarchy import (linkage, cophenet, fcluster)

np.set_printoptions(precision=4)

X = np.array([ [1, 2], [2.5, 4.5], [2, 2], [4, 1.5], [4, 2.5] ])
Y = pdist(X)
print('squareform(Y) = \n', squareform(Y))

Z = linkage(Y)
print('Z = \n', Z)
sch.dendrogram(Z)
plt.show()
plt.close()
print(cophenet(Z, Y)[0])

Y = pdist(X, 'cityblock')
Z = linkage(Y, method='average')
print(cophenet(Z, Y)[0])

T = fcluster(Z, t=2, criterion='maxclust')
print('T = \n', T)

T = fcluster(Z, t=3, criterion='maxclust')
print('T = \n', T)

T = fcluster(Z, t=1.6, criterion='distance')
print('T = \n', T)
