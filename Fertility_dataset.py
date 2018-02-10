import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import ShuffleSplit
import Learning_curves


#Load dataset
Data = pd.read_csv('Dataset/fertility.csv', header=None, index_col=False, names=['season', 'age', 'childish_diseases', 'trauma',
                                                                         'surgical_intervention', 'high_fevers', 'alcohol_consumption',
                                                                         'smoking_habit', 'hours', 'diagnosis'])

#One Hot Encoding
encode_season = pd.get_dummies(Data['season'])
encode_childish_diseases = pd.get_dummies(Data['childish_diseases'])
encode_trauma = pd.get_dummies(Data['trauma'])
encode_surgical_intervention = pd.get_dummies(Data['surgical_intervention'])
encode_high_fevers = pd.get_dummies(Data['high_fevers'])
encode_smoking_habit = pd.get_dummies(Data['smoking_habit'])
encode_diagnosis = pd.get_dummies(Data['diagnosis'])

Data = Data.drop(['season'], axis=1)
Data = Data.drop(['childish_diseases'], axis=1)
Data = Data.drop(['trauma'], axis=1)
Data = Data.drop(['surgical_intervention'], axis=1)
Data = Data.drop(['high_fevers'], axis=1)
Data = Data.drop(['smoking_habit'], axis=1)
Data = Data.drop(['diagnosis'], axis=1)

Attributes_encoded = [Data, encode_season, encode_childish_diseases, encode_trauma, encode_surgical_intervention,
                     encode_high_fevers, encode_smoking_habit, encode_diagnosis]

Data_encoded = pd.concat(Attributes_encoded, axis=1)

#Divide X and y
Data_encoded = np.array(Data_encoded)
X = (Data_encoded[:, 0: (Data_encoded.shape[1]-2)]).astype(float)
y = (Data_encoded[:, -1]).astype(int)

#Predict and plot curves
#cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=0)
BernNB = BernoulliNB()
Perc = Perceptron(max_iter=1000)

title = "Fertility"
curves = Learning_curves.plot_learning_curve(BernNB, Perc, title, X, y, cv=3, n_jobs=1)
curves.show()

