import random
import numpy as np


# Creates a data set of given size
# @param size:
#       The size of the to generate data set
# @param variance:
#       Dictates how much each point in the dataset can vary from the previous point
# @param step:
#       How much to step on average per point, default = 2
# @param correlation:
#       Can be False for no correlation in the data,
#       pos for a positive correlation in the data,
#       neg for a negative correlation in the data
def create_data_set(size, variance, step=2, correlation=False):
    # generate the y values
    val = 1
    ys = []
    for i in range(size):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        if correlation and correlation == 'neg':
            val -= step

    # Appoint an x value for every y value
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
