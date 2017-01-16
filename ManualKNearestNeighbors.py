import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
# We will use warnings to warn about using lower number of K's than we have groups
import warnings
# To count the votes
from collections import Counter
import pandas as pd
import random

# Set the plot style
style.use('fivethirtyeight')

# Create some random data
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
# The features of the example we want to classify in k or r
new_features = [5, 7]


# A function that will return the k nearest neighbors to a given point
# @param data:
#       a dictionary containing the classes and the data for those classes
# @param predict:
#       a vector with the features where for we will make a class prediction
# @param k:
#       the number of nearest neighbors to return, default value 3
# @warning:
#       Throws a warning when the given k is <= the number of elements in the given data set
def k_nearest_neighbors(data, predict, k=3):
    # First create a warning when the number of data points is smaller than or equal to the given k
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    # List with all points and there distances to the prediction
    distances = []
    # For every group calculate the euclidean distance per feature and put it in the distances list
    for group in data:
        for features in data[group]:
            # Calculating the Euclidean Norm, we are using numpy because the calculations are much more efficient
            # than when we would do it manually
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
    # Sort the distances and take the first k elements
    votes = [i[1] for i in sorted(distances)[:k]]
    # Count the votes
    # 1 is the number you want, it returns a list of elements like ('r', 3) with 'r' the name of the class and 3 the
    # number of votes, so we take the first element and then the class name by doing [0][0]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

# Use the k nearest neighbor algorithm to predict the color of the new_features
result = k_nearest_neighbors(dataset, new_features)
# Show us the resulting color
print(result)

# Show our current data
# First manipulate the data a bit and add it in a scatter plot, very nice line btw
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# Throw in the example we want to predict, show the resulting color as well
plt.scatter(new_features[0], new_features[1], s=100, color=result)


# Let's now look at the accuracy on the breast cancer data

# Read our breast cancer data, gathered from UCI:
#       https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
df = pd.read_csv('data/breast-cancer-wisconsin.data')
# Replace the missing values (represented as ?) by -99999
df.replace('?', -99999, inplace=True)
# The ID column is not a good classifier, so we will drop that column
df.drop(['id'], 1, inplace=True)
# Converting the entire data frame to floats
full_data = df.astype(float).values.tolist()

# Shuffle the data
random.shuffle(full_data)
# Split the training and testing data
# Define the test size first
test_size = 0.2
# Define the dictionaries for our test and training data, 2 = benign tumor, 4 = malignant tumors
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
# Split the data in test and training data
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]
# Populate the dictionaries
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

# Train and test the data
# Initialise the total correct predictions to 0 and the total predictions to 0 as well
correct = 0
total = 0

# For every group in our test_set make a prediction using our train_set
for group in test_set:
    for data in test_set[group]:
        # Vote using the default number of k's as defined in Scikit
        vote = k_nearest_neighbors(train_set, data, k=5)
        # If prediction correct, count is as correct
        if group == vote:
            correct += 1
        # Count total
        total += 1

# Show the accuracy result
print('Accuracy', correct/total)


# Show the plot on the initial basic data set
plt.show()
