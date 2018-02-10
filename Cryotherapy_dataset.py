import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
import Learning_curves

#Load dataset
Data = pd.read_csv('Dataset/cryotherapy.csv', header=None, index_col=False, names=['sex', 'age', 'time', 'number_of_Warts',
                                                                         'type', 'area', 'result_of_treatment'])

shuffle(Data)

#One Hot Encoding
encode_sex = pd.get_dummies(Data['sex'])
encode_type = pd.get_dummies(Data['type'])
Data = Data.drop('sex', 1)
Data = Data.drop('type', 1)

#Divide X and y
Data = np.array(Data)
encode_type = np.array(encode_type)
encode_sex = np.array(encode_sex)
X = Data[:, 0: Data.shape[1]-1]
X = np.concatenate((X, encode_type, encode_sex), axis=1)
y = (Data[:, -1]).astype(int)

#Predict and plot curves
BernNB = BernoulliNB()
Perc = Perceptron(max_iter=1000)

title = "Cryotherapy"
curves = Learning_curves.plot_learning_curve(BernNB, Perc, title, X, y, cv=3, n_jobs=1)
curves.show()

