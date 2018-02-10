import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
import Learning_curves

#Load dataset
Data = pd.read_csv('Dataset/mammographic_masses.csv', header=None, index_col=False, names=['bi-rads', 'age', 'shape', 'margin',
                                                                                 'density', 'class'])

# Replace miss data
shuffle(Data)
y = pd.Categorical(Data.pop('class')).codes

Data['bi-rads'].replace({'?': 4}, inplace=True)
Data['age'].replace({'?': 55}, inplace=True)
Data['shape'].replace({'?': 3}, inplace=True)
Data['margin'].replace({'?': 3}, inplace=True)
Data['density'].replace({'?': 3.0}, inplace=True)

#  One Hot Encoding
encode_bi_rads = pd.get_dummies(Data['bi-rads'])
encode_shape = pd.get_dummies(Data['shape'])
encode_margin = pd.get_dummies(Data['margin'])
encode_density = pd.get_dummies(Data['density'])

Data = Data.drop(['bi-rads'], axis=1)
Data = Data.drop(['shape'], axis=1)
Data = Data.drop(['margin'], axis=1)
Data = Data.drop(['density'], axis=1)

Attributes_encoded = [Data, encode_bi_rads, encode_shape, encode_margin, encode_density]
Data_encoded = pd.concat(Attributes_encoded, axis=1)

#Divide X and y
X = Data_encoded.astype(float).values

#Predict and plot curves
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)
BernNB = BernoulliNB()
Perc = Perceptron(max_iter=1000)

title = "Mammographic"
curves = Learning_curves.plot_learning_curve(BernNB, Perc, title, X, y, cv=cv, n_jobs=1)
curves.show()
