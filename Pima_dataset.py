import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import ShuffleSplit
import Learning_curves

#Load dataset
Data = pd.read_csv('Dataset/pima.csv', header=None, index_col=False, names=['pregnant', 'plasma_glucose', 'blood_pressure',
                                                                    'thickness', 'insulin', 'body_mass',
                                                                    'diabetes_pedigree', 'age', 'class'])

#Divide X and y
Data = np.array(Data)
X = (Data[:, 0: (Data.shape[1]-1)]).astype(float)
y = (Data[:, -1]).astype(int)

#Predict and plot curves
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)
BernNB = BernoulliNB()
Perc = Perceptron(max_iter=1000)

title = "Pima"
curves = Learning_curves.plot_learning_curve(BernNB, Perc, title, X, y, cv=cv, n_jobs=1)
curves.show()