# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:50:44 2020

@author: shara
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
crime = pd.read_csv("G:\DS Assignments\Clustering\crime_data.csv")
crime.head()
def norm_func(i) :
    x = (i - i.min())/(i.max() - i.min())
    return x
crime_norm = norm_func(crime.iloc[:, 1:5 ])

k = list(range(2,23))
sse = {}
for i in k :
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_norm)
    crime_norm["clusters"] = kmeans.labels_
    sse[i] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
model = KMeans(n_clusters = 3)
model.fit(crime_norm)
model.labels_
md = pd.Series(model.labels_)
crime_norm["cluster"] = md
crime["cluster"] = md
crime_norm.head()
crime = crime.iloc[ : , [5,0,1,2,3,4]]
cluster = crime.iloc[:,2:6].groupby(crime.cluster).mean()
del cluster1


# Cluster 0 is the most unsafe place to staty, then comes cluster 2 and cluster 1 is the most safest place to stay in all the three. 
# For the female population too, cluster is the most unsafe and cluster 1 is the safest of all.