import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.model_selection import ShuffleSplit
import Learning_curves

#Load dataset
Data = pd.read_csv('Dataset/Adult.csv', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education',
                                                                     'education-num', 'marital-status', 'occupation',
                                                                     'relationship', 'race', 'sex', 'capital-gain',
                                                                     'capital-loss', 'hours-per-week', 'native-country',
                                                                     'class'])

miss_values = np.array(Data)
for i in miss_values:
    for j in range(len(i)):
        if (i[j] == ' ?') :
            i[j] = np.nan

for i in miss_values:
    if i[12] < 39 :
        i[12] = 0
    else:
        i[12] = 1

Data = pd.DataFrame(miss_values)
Data.columns = ['age', 'workclass', 'fnlwgt', 'education',
                 'education-num', 'marital-status', 'occupation',
                 'relationship', 'race', 'sex', 'capital-gain',
                 'capital-loss', 'hours-per-week', 'native-country',
                 'class']
Data = Data.dropna(how='any')

y = pd.Categorical(Data.pop('class')).codes

Data['marital-status'].replace({' Married-civ-spouse': ' Married'}, inplace=True)
Data['marital-status'].replace({' Divorced': ' Not-Married'}, inplace=True)
Data['marital-status'].replace({' Never-married': ' Not-Married'}, inplace=True)
Data['marital-status'].replace({' Separated': ' Not-Married'}, inplace=True)
Data['marital-status'].replace({' Widowed': ' Not-Married'}, inplace=True)
Data['marital-status'].replace({' Married-spouse-absent': ' Married'}, inplace=True)
Data['marital-status'].replace({' Married-AF-spouse': ' Married'}, inplace=True)

#One Hot Encoding
encode_workclass = pd.get_dummies(Data['workclass'])
encode_education = pd.get_dummies(Data['education'])
encode_marital = pd.get_dummies(Data['marital-status'])
encode_occupation = pd.get_dummies(Data['occupation'])
#encode_relationship = pd.get_dummies(Data['relationship'])
encode_race = pd.get_dummies(Data['race'])
encode_sex = pd.get_dummies(Data['sex'])
#encode_country = pd.get_dummies(Data['native-country'])

Data = Data.drop(['workclass'], axis=1)
Data = Data.drop(['education'], axis=1)
Data = Data.drop(['education-num'], axis=1)
Data = Data.drop(['marital-status'], axis=1)
Data = Data.drop(['occupation'], axis=1)
Data = Data.drop(['relationship'], axis=1)
Data = Data.drop(['race'], axis=1)
Data = Data.drop(['sex'], axis=1)
Data = Data.drop(['native-country'], axis=1)
Data = Data.drop(['fnlwgt'], axis=1)
#Data = Data.drop(['hours-per-week'], axis=1)

Attributes_encoded = [Data, encode_workclass, encode_sex, encode_marital, encode_occupation, encode_race]
Data_encoded = pd.concat(Attributes_encoded, axis=1)

X = Data_encoded.astype(float).values

#Predict and plot curves
cv = ShuffleSplit(n_splits=10, test_size=.25, random_state=0)
BernNB = MultinomialNB()
Perc = Perceptron(max_iter=1000)

title = "Adult"
curves = Learning_curves.plot_learning_curve(BernNB, Perc, title, X, y, cv=cv, n_jobs=1)
curves.show()

