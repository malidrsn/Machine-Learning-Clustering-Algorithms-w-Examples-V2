# K değerine karar vermek önemlidir.
# tüm data pointler aynıdır.
# dirsek noktası K değeri olarak atanır. wcss önemlidir.

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
