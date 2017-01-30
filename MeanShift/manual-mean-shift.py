import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

# Our data set
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9,3]])

# Used colors
colors = 10*["g", "r", "c", "b", "k"]


# The Mean Shift class
class MeanShift:

    # @var radius
    #       The radius for our Mean Shift algorithm, our algorithm will find this by itself
    # @var radius_norm_step
    #       The step with which the radius increases.
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    # Fits the given data to the classifier
    def fit(self, data):
        # If the radius is not defined, we will find it ourselves
        if not self.radius:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        # Initialise a dictionary for our centroids
        centroids = {}

        # the initial centroids are all our data points
        for i in range(len(data)):
            centroids[i] = data[i]

        # our weights for changing the radius
        weights = [i for i in range(self.radius_norm_step)][::-1]
        # Optimise the centroids
        while True:
            new_centroids = []
            # For every centroid, find every all feature sets within it's radius and calculate the new
            # centroid based on the mean of these feature sets.
            for i in centroids:
                in_radius = []
                centroid = centroids[i]
                # Go over every feature set to see if it's in the current centroid's radius, possibly change the radius
                # Let every feature set vote for this change in radius, by using their representative weights
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.0000000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_radius += to_add

                # Calculate the new centroid based on all feature sets in the radius
                new_centroid = np.average(in_radius, axis=0)
                # and add it as a tuple to the new centroids
                new_centroids.append(tuple(new_centroid))
            # Continue only with the unique new centroids
            uniques = sorted(list(set(new_centroids)))

            # Because our radius is changing, it is possible that centroids are very close to eachother,
            # We want to merge these:
            to_pop = []
            for i in uniques:
                for ii in [i for i in uniques]:
                    if i == ii:
                        pass
                    # Check if the given centroid is within the radius of another centroid
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break
            # Remove the centroids that are too close to other centroids
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            # Remember the previous centroids
            prev_centroids = dict(centroids)

            # Set the unique new centroids as the current centroids
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            # Check if optimised, only optimised when the current centroids are equal to the previous ones
            optimised = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimised = False
                if not optimised:
                    break

            # If optimised, break
            if optimised:
                break

        # Set the centroids of this class to the calculated centroids
        self.centroids = centroids

        # Classify our data as well on the found centroids
        self.classifications = {}

        # Add a place in the dictionary for every centroid
        for i in range(len(self.centroids)):
            self.classifications[i] = []

        # Classify every feature set in the data
        for featureset in data:
            # Compare distance to all centroids
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            # The classification is the centroid that is closest
            classification = (distances.index(min(distances)))

            # add the classification
            self.classifications[classification].append(featureset)

    # Predicts the class of a given feature set
    def predict(self, data):
        # Compare distance to all centroids
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        # The classification is the centroid that is closest
        classification = (distances.index(min(distances)))
        return classification


# Create the classifier
clf = MeanShift()
# Fit the data
clf.fit(X)
# Get our calculated centroids
centroids = clf.centroids

# Visualise
# add the centroids to the plot
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

# Add all classifications to the plot
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5, zorder=10)
# add the original feature sets to the plot
plt.scatter(X[:, 0], X[:, 1])

# show the plot
plt.show()
