import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np


def plot_learning_curve(NaiveBayes, Perceptron,  title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1., 10)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    train_sizesNB, train_scoresNB, test_scoresNB = learning_curve(NaiveBayes, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_sizesP, train_scoresP, test_scoresP = learning_curve(Perceptron, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    # Naive Bayes
    train_scores_meanNB = 1 - np.mean(train_scoresNB, axis=1)
    train_scores_stdNB = 1 - np.std(train_scoresNB, axis=1)
    test_scores_meanNB = 1 - np.mean(test_scoresNB, axis=1)
    test_scores_stdNB = 1 - np.std(test_scoresNB, axis=1)

    # Perceptron
    train_scores_meanP = 1 - np.mean(train_scoresP, axis=1)
    train_scores_stdP = 1 - np.std(train_scoresP, axis=1)
    test_scores_meanP = 1 - np.mean(test_scoresP, axis=1)
    test_scores_stdP = 1 - np.std(test_scoresP, axis=1)
    plt.grid()

    plt.plot(train_sizesP, test_scores_meanP, 'o-', color="g", label="Perceptron")
    plt.plot(train_sizesNB, test_scores_meanNB, 'o-', color="b", label="Naive Bayes")

    plt.legend(loc="best")
    return plt