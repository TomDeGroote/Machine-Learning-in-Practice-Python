from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

style.use('ggplot')

# The centers for the generated blob data
centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
# Our feature sets, generate data using a generator
X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=1.5)

# Mean Shift classifier
ms = MeanShift()
# Fit the feature sets
ms.fit(X)
# Get the assigned labels
labels = ms.labels_
# Get the cluster centers
cluster_centers = ms.cluster_centers_
sorted(cluster_centers, key=lambda x: x[0])

# Show the predicted centers vs the original ones
print(len(centers), "original centers for data generation:", centers)
print(len(cluster_centers), "predicted centers by Mean Shift:", cluster_centers)

# Initialise the 3D plot
colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add the feature sets to the plot
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

# Add the cluster centers to the plot
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', color='k', s=150,
           linewidths=5, zorder=10)

# Show the plot
plt.show()
