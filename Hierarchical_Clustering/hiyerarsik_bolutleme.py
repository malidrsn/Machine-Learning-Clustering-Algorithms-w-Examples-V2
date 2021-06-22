# agglomerative = asagidan yukariya sistem calismasi yaklasimi n cluster -> 1 cluster
# divisive = yukaridan asagiya sistem calismasi yaklasimi 1 cluster -> n cluester


# kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
data = pd.read_csv("musteriler.csv")
# print(data)

X = data.iloc[:, 3:].values

# K-means algorithm
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, init="k-means++")
kmeans.fit(X)
print(kmeans.cluster_centers_)  # merkez noktalarını veriyor

# k için optimum değeri bulalım wcss ile
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia = wcss

plt.plot(range(1, 11), wcss)
plt.show()

kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
y_tahmin = kmeans.fit_predict(X)
print("y-tahmin", y_tahmin)
plt.scatter(X[y_tahmin == 0, 0], X[y_tahmin == 0, 1], s=100, c="red")
plt.scatter(X[y_tahmin == 1, 0], X[y_tahmin == 1, 1], s=100, c="blue")
plt.scatter(X[y_tahmin == 2, 0], X[y_tahmin == 2, 1], s=100, c="green")
plt.scatter(X[y_tahmin == 3, 0], X[y_tahmin == 3, 1], s=100, c="yellow")
plt.title("K-Means")
plt.show()

print("**************** HC ****************")
# Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering

agg_cls = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")

y_tahmin = agg_cls.fit_predict(X)
print("y-tahmin", y_tahmin)

plt.scatter(X[y_tahmin == 0, 0],
            X[y_tahmin == 0, 1], s=100, c="red")  # x datasında ki y_tahminin 0 olanların 0. kolonu ve 1.kolonu demek
plt.scatter(X[y_tahmin == 1, 0], X[y_tahmin == 1, 1], s=100, c="blue")
plt.scatter(X[y_tahmin == 2, 0], X[y_tahmin == 2, 1], s=100, c="green")
plt.scatter(X[y_tahmin == 3, 0], X[y_tahmin == 3, 1], s=100, c="yellow")
plt.title("HC")
plt.show()

# dendrogram
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))

plt.show()
# dendrogram'a bakarak 2-4 alınabilmektedir.
