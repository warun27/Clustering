# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 05:25:55 2020

@author: shara
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import excel
airlines = pd.read_excel("G:\DS Assignments\Clustering\EastWestAirlines.xlsx", index = None, header = 0)
type(airlines)
pd.DataFrame(airlines)
airlines = pd.read_excel("G:\DS Assignments\Clustering\EastWestAirlines.xlsx", sheet_name="data")
airlines.dtypes
def norm_func(i) :
    x = (i - i.min())/(i.max()) - (i.min())
    return x
airlines_norm = norm_func(airlines.iloc[:,1:10])
airlines_norm.head()
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(airlines_norm, method = "complete" , metric = "euclidean")
plt.figure(figsize = (15,5)) ; plt.title("H_Clustering_Dendrogram");plt.xlabel("index");plt.ylabel("Distance")
sch.dendrogram(z)
plt.show()
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete',affinity = "euclidean").fit(airlines_norm)
cluster_labels  = pd.Series(h_complete.labels_)
airlines["Cluster"] = cluster_labels
airlines = airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines.Cluster.value_counts()
airlines.iloc[:,2:13].groupby(airlines.Cluster).mean()
airlines_C_average = airlines.iloc[:,2:12].groupby(airlines.Cluster).mean()
awards = airlines.iloc[:,12].groupby(airlines.Cluster).sum()
airlines_C_average["Awards"] = awards

# Cluster 0 clients have the least demand for flights and travel the least, giving them so many free flights is not encouraging them to fly more as thir demand of trabel is less, the free flight system here should be replaces with discounts on the flights booked and not give unnecessary free flights.
# Cluster 1 clients do travle a lot and have the highest number of flights booked in the past 12 months, free flights or awards can be increased to encourage them to fly with the airlines.
# Cluster 2 customers are the highest users of the frequent flier credit card and are the oldest customers. These customers should be given a an between the free flight ticket award or discount system to chose from.