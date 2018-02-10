import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
import Learning_curves


# Load dataset
Data = pd.read_csv('Dataset/sonar.csv', header=None, index_col=False)
Data.replace({'R': 0, 'M': 1}, inplace=True)

# Discrete values
Data = np.array(Data)

for i in Data:
    for j in range(len(i)):
        if i[j] < 0.25:
            i[j] = 1
        elif i[j] >= 0.25 and i[j] < 0.5:
            i[j] = 2
        elif i[j] >= 0.5 and i[j] < 0.75:
            i[j] = 3
        else:
            i[j] = 4

Data = pd.DataFrame(Data)
Encoded_data = pd.DataFrame()

# One Hot Encoding
for i in range(Data.shape[1]):
    Encoded = pd.get_dummies(Data[i])
    Encoded_data = pd.concat([Encoded_data, Encoded], axis=1)

Data = np.array(Encoded_data)
shuffle(Data)

# Divide X and y
X = (Data[:, 1: (Data.shape[1]-1)]).astype(float)
y = (Data[:, -1]).astype(int)

#Predict and plot curves
cv = ShuffleSplit(n_splits=100, test_size=.25, random_state=None)
BernNB = BernoulliNB()
Perc = Perceptron(max_iter=1000)

title = "Sonar"
curves = Learning_curves.plot_learning_curve(BernNB, Perc, title, X, y, cv=cv, n_jobs=1)
curves.show()





