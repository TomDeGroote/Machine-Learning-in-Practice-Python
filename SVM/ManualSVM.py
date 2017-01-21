import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

# Our plot style
style.use('ggplot')

# Our starting data
# As you can see we have two classes, -1 and 1 with some data for those classes
data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8], ]),
             1: np.array([[5, 1], [6, -1], [7, 3], ])}


# A class definition of a SVM
# The class doesn't explain all the details of how an SVM works, for more information
# look at svm-tutorial.com
class SupportVectorMachine:

    # Runs whenever an object is created
    def __init__(self, visualisation=True):
        # Set the object variable visualisation to the given value
        self.visualisation = visualisation
        # Set the colors for the two classes
        self.colors = {1: 'r', -1: 'b'}

        # Visualise if necessary
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # A method that visualises the SVM
    def visualise(self):
        # scattering known featuresets
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # Returns the hyperplane point given x, w, b and v
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        # Some useful variables
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # Graph the positive support vector hyperplane
        # w.x + b = 1
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], "k")

        # Graph the negative support vector hyperplane
        # w.x + b = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], "k")

        # Graph the decision surface
        # w.x + b = 0
        ds1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        ds2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [ds1, ds2], "k")

        # show the plot
        plt.show()

    # Predict the class given a set of features
    def predict(self, features):
        # Find the classification given a set of features by looking at the sign of the
        # following formula: sign(x.w +b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        # For visualisation puposes:
        if classification != 0 and self.visualisation:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        elif self.visualisation:
            print('featureset', features, 'is on the decision boundary')

        return classification

    # train the SVM
    def fit(self, data):
        self.data = data

        # Begin building an optimisation dictionary, which contains any optimisation values
        # When all optimisations are done we will pick the [w, b] value with the lowest ||w||
        # in this dictionary
        #
        # The optimisation is done by stepping down the vector w and calculating the largest b that will
        # fit the x.w + b equation. The found values will be saved in this dictionary
        #
        # How the dictionary will look: {||w||: [w, b]}
        opt_dict = {}

        # This enables us to check every version of the vector possible
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        # Pick a starting point that fits our data
        # First put all features in all_data
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        # Then pick the starting point
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        # We don't need all data anymore so can throw it out of memory
        all_data = None

        # For support vectors yi(xi.w + b) = 1, we will now start looking for this by searching w and b that fit the
        # bill

        # Pick step_sizes for stepping down the vector, we start with big steps,
        # when find the minimum for this step size, we start taking smaller steps to find a better minimum.
        # By not starting with the smallest steps we save a lot of calculating power.
        # This is one way to approach the optimisation problem. With these steps we will approach w.
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]

        # With these steps we will approach b
        # You could implement a similar step size feature for finding an accurate b the same way as above
        # But for brevity this was skipped. (Might be a TODO for later)
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        # Now we are ready to approach w and b by stepping
        for step in step_sizes:
            # Initial pick for w, we can do this because it is a convex problem
            w = np.array([latest_optimum, latest_optimum])

            # Step down the convex bowl until you find the optimal point with this step size
            optimized = False
            while not optimized:
                # Iterate through possible b values
                # Try every b from -max_feature_value * b_range_multiple to +... with a step size of
                # step * b_multiple
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    # Check the equation x.w + b for every possible transformation
                    for transformation in transforms:
                        # Transform w
                        w_t = w * transformation
                        #
                        found_option = True
                        # Implement the constraint: yi(xi.w + b) >= 1
                        # Weakest link of SVM, SMO tries to fix this
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                # Check if constraint fulfilled
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break  # TODO als het niet werkt kom eens hier kijken
                            if not found_option:
                                break  # TODO als het niet werkt kom eens hier kijken
                        # If constrained fulfilled then add the option to our optimisation dictionary
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # Check if the first element of the vector w is < 0,  we can stop searching, because we
                # already check the w's < 0  with the transformations
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    # Step on w to approximate the optimal value better
                    w = w - step

            # We found an optimum, so let's see what our best possible w and b are. For that we need to select
            # The w and b where the magnitude (aka the norm) of w is minimal
            # Calculate all the norms
            norms = sorted([n for n in opt_dict])
            # Our optimal choice
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # Redefine our latest optimum
            latest_optimum = opt_choice[0][0] + step*2


# Create an SVM object
svm = SupportVectorMachine()
# Make our data fit
svm.fit(data=data_dict)

# Values to predict
predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]

# Make predictions
for p in predict_us:
    svm.predict(p)

# Show us the magic
svm.visualise()
