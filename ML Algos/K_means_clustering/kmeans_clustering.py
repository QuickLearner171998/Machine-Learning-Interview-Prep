# -*- coding: utf-8 -*-
"""
Ref - https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25

# Kmeans Algo

Steps -
1. Select Value of K
2. Randomly Select K datapoints to be centroids initially
3. Compute dist of all other points from the K centroids
4. Assign each point into one of the centroids (closest dist)
5. Calculate mean coords of each cluster to get new centroids
6. Repeat 3,4,5 until convergence
"""

import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
import sys
# Prepare data
titanic = pd.read_csv('titanic.csv')

cluster_data = titanic[['Fare', 'Age']].copy(deep=True)
cluster_data.dropna(axis=0, inplace=True)
cluster_data.sort_values(by=['Fare', 'Age'], inplace=True)

cluster_array = np.array(cluster_data)

# Helper Func


def calcDist(a, b):
    return (sum((a-b)**2))**0.5

# Assign cluster clusters based on closest centroid
def assign_clusters(centroids, cluster_array):
    # Here aim is to create a clusters arr where
    # clusters[i] will tell to which cluster cluster_array[i] belongs

    clusters = []
    for i, data in enumerate(cluster_array):
        min_dist = sys.maxsize
        min_ind = 0
        for j, centroid in enumerate(centroids):
            d = calcDist(centroid, data)
            if d < min_dist:
                min_dist = d
                min_ind = j
        clusters.append(min_ind)
    return clusters


def calc_cluster_mean(i, clusters, cluster_array):
    cluster_df = pd.concat([pd.DataFrame(cluster_array),
                            pd.DataFrame(clusters, columns=['cluster'])], axis=1)
    cluster_col = cluster_df['cluster']
    sub_df = cluster_df[cluster_col == i][cluster_df.columns[:-1]]
    c1, c2 = sub_df.mean(axis=0)
    return np.array((c1, c2)), sub_df


def update_centroids(k, clusters, cluster_array):
    new_centroid = []
    for i in range(k):
        mean, _ = calc_cluster_mean(i, clusters, cluster_array)
        new_centroid.append(mean)
    return new_centroid


def cluster_variance(k, clusters, cluster_array):
    # var = sum(x-mu) ** 2
    cluster_var = []
    for i in range(k):
        cluster_mean, sub_cluster = calc_cluster_mean(i, clusters, cluster_array)
        # make cluster mean at each index for easy computation of variance
        # repmat
        cluster_mean_rep = np.matlib.repmat(cluster_mean, len(sub_cluster), 1)
        var = np.sum(np.sum((sub_cluster-cluster_mean_rep)**2))
        cluster_var.append(var)
    return cluster_var


# main code
k = 4
iter_var = []
centroids = [cluster_array[i+2] for i in range(k)]
clusters = assign_clusters(centroids, cluster_array)
initial_clusters = clusters
print("Cluster Var", round(np.mean(cluster_variance(k, clusters, cluster_array))))

# train
for i in range(20):
    centroids = update_centroids(k, clusters, cluster_array)
    clusters = assign_clusters(centroids, cluster_array)
    cluster_var = np.mean(cluster_variance(k, clusters, cluster_array))
    iter_var.append(cluster_var)
    print(i+1, round(cluster_var))

# Plot Var curve
plt.subplots(figsize=(9, 6))
plt.plot(iter_var)
plt.xlabel('Iterations')
plt.ylabel('Mean Sum of Squared Deviations')

# Plot clusters Initial
plt.subplots(figsize=(9, 6))
plt.scatter(x=cluster_array[:, 0],
            y=cluster_array[:, 1],
            c=initial_clusters,
            cmap=plt.cm.Spectral)
plt.xlabel('Passenger Fare')
plt.ylabel('Passenger Age')

# Plot clusters Initial
plt.subplots(figsize=(9, 6))
plt.scatter(x=cluster_array[:, 0],
            y=cluster_array[:, 1],
            c=clusters,
            cmap=plt.cm.Spectral)
plt.xlabel('Passenger Fare')
plt.ylabel('Passenger Age')
