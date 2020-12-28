# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:19:48 2020

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
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(crime_norm, method = "complete" , metric = "euclidean")
plt.figure(figsize = (15,5)) ; plt.title("H_Clustering_Dendrogram");plt.xlabel("index");plt.ylabel("Distance")
sch.dendrogram(z)
plt.show()
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete',affinity = "euclidean").fit(crime_norm)
cluster_labels  = pd.Series(h_complete.labels_)
crime_norm["Cluster"] = cluster_labels
crime["Cluster"] = cluster_labels
crime = crime.iloc[ : , [5,0,1,2,3,4]]
cluster = crime.iloc[:,2:6].groupby(crime.Cluster).mean()


# Custer 2 states seems to be the safest places of all the rest clusters. It is also the safest place for the females too.
# Cluster 1 state comes after the cluster 2 as the population is also high and crime rate in comparision with Cluster 0 and 3 is the least.
# Cluster 0  is the most unsafe state for the population. for the current population in the cluster, it seems like crime is very common, especially rape and assault cases.
# Cluster 3 also not safe to stay however is safer than cluster 0. Cluster 3 can be determined as the most unsafe place for the female population.