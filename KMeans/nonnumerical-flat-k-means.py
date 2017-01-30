import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd

# Read the excel file which contains the data
df = pd.read_excel('../data/titanic.xls')
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

# Since using flat clustering we need to tell it how many clusters to find.
# We are going to look for groups of survivors and non-survivors

# If you want to see how much of an impact individual features have on the end result,
# You can uncomment these lines. If the accuracy drops significantly,
# it has a large impact on the results.
# df.drop(['boat'], 1, inplace=True)
# df.drop(['sex'], 1, inplace=True)

# Our X data (without the survivor feature)
X = np.array(df.drop(['survived'], 1).astype(float))
# Preprocess our X, aka scale it to range between -1 and 1
X = preprocessing.scale(X)
# Our y data, aka the survivor feature
y = np.array(df['survived'])

# Define our classifier
clf = KMeans(n_clusters=2)
# Fit the data
clf.fit(X)

# Let's check our accuracy, for us survivors are 1 and non-survivors are 0, but clustering
# uses random labels, so our accuracy is whatever is highest, since with clustering non-survivors
# could be labeled 1 and survivors 1.
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

# Calculate the accuracy of the predictions
accuracy = (correct/len(X))
print("Accuracy:", accuracy)
