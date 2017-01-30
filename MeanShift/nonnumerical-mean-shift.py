import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn import preprocessing

# When running you can ignore the warning, the program takes a while to make it's calculations, so don't worry.
# Read the excel file which contains the data
df = pd.read_excel('../data/titanic.xls')
original_df = pd.DataFrame.copy(df)
# Drop unimportant columns
df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)


# Convert the non-numerical data in the data set to numerical data
def handle_non_numerical_data(df):
    columns = df.columns.values
    # Go through every column
    for column in columns:
        # Initialise a dictionary for our numerical value of a label
        text_digit_vals = {}

        # Function to convert a textual label to its numerical representation
        def convert_to_int(val):
            return text_digit_vals[val]

        # If the values of a column are nonnumerical, convert them
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            # The contents of a column
            column_contents = df[column].values.tolist()
            # The unique elements of a column
            unique_elements = set(column_contents)
            # Add all unique elements to the dictionary
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            # Convert column from their nonnumerical representation to their numerical representation
            df[column] = list(map(convert_to_int, df[column]))
    return df

# Convert the nonnumerical data to numerical data
df = handle_non_numerical_data(df)

# Our X data (without the survivor feature)
X = np.array(df.drop(['survived'], 1).astype(float))
# Pre-process our feature sets, aka scale it to range between -1 and 1
X = preprocessing.scale(X)
# Our y data, aka the survivor feature
y = np.array(df['survived'])

# Define our classifier
clf = MeanShift()
# Fit the data
clf.fit(X)

# Get the labels from our classifier
labels = clf.labels_
# Get the cluster centers from our classifier
cluster_centers = clf.cluster_centers_

# Add a cluster group column to our original data set (This is to view what kind of clusters were generated)
original_df['cluster_group'] = np.nan
# populate the new column
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

# Check the survival rate of each of the found groups
# First get the number of clusters
n_clusters_ = len(np.unique(labels))
# initialise a dictionary for the survival_rates of the clusters
survival_rates = {}
# Add the survival rates of the clusters to the dictionary
for i in range(n_clusters_):
    # Get all the rows in our original df where the cluster group equals the current cluster group i
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    # Get all the survivals in the cluster
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    # Calcualte survival rate
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

# Show us the survival rates, the number of groups and survival rates can differ every time since some
# randomness is involved
print(survival_rates)
