import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_confusion_matrix(test_OUTPUT, predicted_OUTPUT):

    # confusion matrix

    cnf_matrix = metrics.confusion_matrix(test_OUTPUT, predicted_OUTPUT)
    print("confusion matrix: ")
    print(cnf_matrix)
    return cnf_matrix


def plot_confusion_matrix(cnf_matrix):

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
