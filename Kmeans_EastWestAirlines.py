# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:52:25 2020

@author: shara
"""
# KMeans Clustering
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
airlines = pd.read_excel("G:\DS Assignments\Clustering\EastWestAirlines.xlsx", sheet_name = "data")
airlines.head()
def norm_func(i) :
    x = (i - i.min())/(i.max() - i.min())
    return x
airlines_norm = norm_func(airlines.iloc[:, 1:11 ])

# The greyed out portion doesnt work
# k = list(range(2,23))
# twss = {}
# for i in k :
#     kmeans = KMeans(n_clusters = i)
#     kmeans.fit(airlines_norm)
#     wss = {}
#     for j in range(i):
#         wss.append(sum(cdist(airlines_norm.iloc[kmeans.labels_ == j,:], kmeans.cluster_centers_[j].reshape(1, airlines_norm.shape[1]), "euclidean")))
#         twss.append(sum(wss))

# k_a = np.array(k)
# twss_a = np.array(twss)    
# plt.plot(list(k),list(twss),"ro-");plt.xlabel("No_of_clusters");plt.ylabel("Total_within_sum_of_sqares");plt.xticks(k_a)

k = list(range(2,23))
sse = {}
for i in k :
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    airlines_norm["clusters"] = kmeans.labels_
    sse[i] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

model = KMeans(n_clusters = 4)
model.fit(airlines_norm)
model.labels_
md = pd.Series(model.labels_)
airlines_norm["cluster"] = md
airlines["cluster"] = md
airlines_norm.head()
clusters1 = airlines.iloc[:,[1,2,3,4,5,6,7,8,9,10]].groupby(airlines.cluster).mean()

Cluster 3 clients has the highest usage of the airline flight, they also spend a lot in non flight transactions. this group can be classified as the highest earning group.
Cluster 2 clients doesnt prefer to use the credit card at every ocassion, they travle the least too, this group can be classified as the least earning group.
Cluster 1 customers do not travel a lot, however the credit card usage outside the flights transactions is the highest. This group can be classified as above the average earning group or slightly below the higher earning group.
Cluster 0 customers are the average earning group and a good balace between expenses on trabel and non flight transactions can be seen.

Cluster 3 customer should be give the awards to mantain these customers for a longer period.
Cluster 1 should also be given the free flight award to encourage customer fly more on a frequent basis.
Cluster 0 and 2 should be given discounts on flights to ensure long time enrollment with the airlines. Giving them the free flights wont be much benefical as the demand of them travelling seems to be quiet low.