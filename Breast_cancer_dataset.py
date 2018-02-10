import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import ShuffleSplit
import Learning_curves

#Load dataset
Data = pd.read_csv('Dataset/Breast_cancer.csv')

#Cast from string to float, and miss values
Data.replace({'?': 0}, inplace=True)
Data = np.array(Data)
for i in range(Data.shape[1]):
    toparse = Data[:, i]
    result = [float(j) for j in toparse]
    Data[:, i] = result

Data = pd.DataFrame(Data)
Data.columns = ['code', 'Thickness', 'Cell-Size', 'Cell Shape', 'Adhesion', 'Epithelial-Cell-Size', 'Bare_Nuclei',
                'Bland Chromatin', 'Normal_Nucleoli', 'Mitoses', 'class']

Data = Data.drop(['code'], axis=1)
Data['Bare_Nuclei'].replace({0: 3}, inplace=True)

#Divide X and y
y = pd.Categorical(Data.pop('class')).codes
X = Data.astype(float).values

#Predict and plot curves
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=None)
BernNB = BernoulliNB(alpha=0.1)
Mult = MultinomialNB(alpha=0.1)
Perc = Perceptron(max_iter=1000)

title = "Breast Cancer"
curves = Learning_curves.plot_learning_curve(Mult, Perc, title, X, y, cv=cv, n_jobs=4)
curves.show()

