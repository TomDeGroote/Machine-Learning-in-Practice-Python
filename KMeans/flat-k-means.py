import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')

# Our data set
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# Fit the data
clf = KMeans(n_clusters=2)
clf.fit(X)

# Grab the centroids
centroids = clf.cluster_centers_
# Grab the labels
labels = clf.labels_

# visualise
colors = ["g.", "r.", "c.", "y."]
# Color the data set according to the given labels by the classifier
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
# Also show the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()
