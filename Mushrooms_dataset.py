import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import ShuffleSplit
import Learning_curves

#Load dataset
Data = pd.read_csv('Dataset/mushrooms.csv', header=None, index_col=False, names=['edibile', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                                                                         'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                                                                         'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                                                                         'stalk-surface-below-ring', 'stalk-color-above-ring',
                                                                         'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                                                                         'ring-type', 'spore-print-color', 'population', 'habitat'])

#One Hot Encodind
Data_dummies = pd.get_dummies(Data)
Data_encoded = np.array(Data_dummies)
np.random.shuffle(Data_encoded)

#Divide X and y
X = (Data_encoded[:, 2: Data_encoded.shape[1]]).astype(float)
y = Data_encoded[:, 0]

#Predict and plot curves
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)
BernNB = BernoulliNB()
Perc = Perceptron(max_iter=1000)

title = "Mushrooms"
curves = Learning_curves.plot_learning_curve(BernNB, Perc, title, X, y, cv=cv, n_jobs=1)
curves.show()