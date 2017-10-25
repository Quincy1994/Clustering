## coding=utf-8

print(__doc__)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state,cluster_std=[0.5,0.5,0.5])

# 预测
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)


## 可视化
plt.plot()
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Example of Clustering Blobs by K-means")
plt.show()
