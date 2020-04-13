import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as cnf_plt
import numpy as np
import pandas as pd


def get_confusion_matrix(test_OUTPUT, predicted_OUTPUT):

    cnf_matrix = metrics.confusion_matrix(test_OUTPUT, predicted_OUTPUT)
    print("confusion matrix: ")
    print(cnf_matrix)
    return cnf_matrix


def plot_confusion_matrix(cnf_matrix):

    class_names = [0, 1]  # name  of classes
    fig, ax = cnf_plt.subplots()
    tick_marks = np.arange(len(class_names))
    cnf_plt.xticks(tick_marks, class_names, fontsize=14)
    cnf_plt.yticks(tick_marks, class_names, fontsize=14)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 22}, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    cnf_plt.tight_layout()
    cnf_plt.title('Confusion Matrix', y=1.1, fontsize=14)
    cnf_plt.ylabel('Actual label', fontsize=14)
    cnf_plt.xlabel('Predicted label', fontsize=14)
    conf_matrix_fig = cnf_plt.gcf()
    conf_matrix_fig.tight_layout()
    cnf_plt.draw()
    conf_matrix_fig.savefig('plots/conf_matrix_plt.png')
    cnf_plt.clf()
