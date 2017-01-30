import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

# Our data set
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])


class KMeans:
    # @var k
    #       The number of clusters
    # @var tol
    #       The tolerance for optimisation
    # @var max_iter
    #       The maximum number of iterations to run for the optimisation.
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    # This method fits the given data to the model
    def fit(self, data):
        # Set the initial centroids, to the first k elements of the given data set
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        # For a maximum nr of iterations optimise the data classification
        for i in range(self.max_iter):
            # Initialising classification dictionary
            self.classifications = {}
            for j in range(self.k):
                self.classifications[j] = []

            # Do the actual classification based on the distance of the data point to the centroids
            for featureset in data:
                # The distance to all centroids of a given feature set
                distances = [np.linalg.norm(featureset-self.centroids[cen]) for cen in self.centroids]
                # Classify the feature set to the class of the centroid with the smallest distance to the feature set
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            # Remember the previous centroids to check if we fulfill the tolerance requirement
            prev_centroids = dict(self.centroids)
            # Find the new centroids
            for classification in self.classifications:
                # The new centroids are the averages of all the feature sets in the classifcation
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # Check if currently optimised
            optimised = True
            for c in self.centroids:
                # Previous centroid
                original_centroid = prev_centroids[c]
                # Current centroid
                current_centroid = self.centroids[c]
                # If it moved more than the tolerance, show how much it moved and set optimised to false.
                if np.sum((current_centroid-original_centroid)/original_centroid*100) > self.tol:
                    optimised = False
            # If we fulfill the requirements, we are optimised and can stop optimising
            if optimised:
                break

    # Predicts the class of the given feature set
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# Define our classifier
clf = KMeans()
# Fit our data
clf.fit(X)

# visualise
colors = 10*["g", "r", "c", "b", "k"]
# Add the centroids to the plot
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)

# Add all data to the plot, with their classification
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

# Let's see how well we perform for unknown data
unknowns = np.array([[1, 3], [8, 9], [0, 3], [5, 4], [6, 4]])
# ask a prediction for the unknowns and plot them
for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], color=colors[classification], marker="*", s=150, linewidths=5)

# plot
plt.show()