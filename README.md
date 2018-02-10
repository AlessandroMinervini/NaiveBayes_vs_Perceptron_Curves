# NaiveBayes_vs_Perceptron_Curves

Plot of 'Generalization Error vs. Training Examples' on Naive Bayes and Perceptron predictions

## Run
1. Download the [repository](https://github.com/AlessandroMinervini/NaiveBayes_vs_Perceptron_Curves), the dataset are included in 'Dataset' folder.

2. For each dataset run /dataset_name.py to plot the respective error curve.

## Used Datasets
From [UCI Repository](http://archive.ics.uci.edu/ml/index.php):
- Adult
- Blood Trasfusion
- Breast Cancer
- Cryotherapy
- Fertility
- Ionosphere
- Mammographic Masses
- Mushrooms
- Pima
- Sonar

## Implementation
Dependently on each kind of dataset, some pre-processing operations have been done.
The method used for classification is Cross Validation, 4-fold or stratified 3-fold.
The function *plot_learning_curve()* determines cross-validated test scores for different training set sizes, and plots the Naive Bayes and Perceptron curves.

## Used libraries
- Numpy
- Pandas
- Sklearn
- Matplotlib