import numpy as np
import pandas as pd
from som import SOM
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

# read in data
boston = load_boston().data
# standard scale data
ss = StandardScaler()
X = ss.fit_transform(boston)

# instantiate SOM
som = SOM(X, 3, 3, 1, 100, 0.01)
# train model
weights, winners, distances = som.train(X)
# print data
print(weights)
