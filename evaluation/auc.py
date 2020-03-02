from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as auc_plt
import numpy as np
import pandas as pd
import pylab as pl


def plot_AUC(test_OUTPUT, prediction_probs):

    # no skill prediction
    no_skill_probs = [0 for _ in range(len(test_OUTPUT))]

    # calculate scores
    no_skill_auc = roc_auc_score(test_OUTPUT, no_skill_probs)
    model_auc = roc_auc_score(test_OUTPUT, prediction_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % no_skill_auc)
    print('Logistic: ROC AUC=%.3f' % model_auc)

    # calculate roc curves
    no_skills_false_pos_rate, no_skill_true_pos_rate, no_skill_thresholds = roc_curve(test_OUTPUT, no_skill_probs)
    model_false_positive_rate, model_true_pos_rate, model_thresholds = roc_curve(test_OUTPUT, prediction_probs)

    # plot the roc curve for the model
    auc_plt.plot(no_skills_false_pos_rate, no_skill_true_pos_rate, linestyle='--', label='No skills')
    auc_plt.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='Logistic')

    # axis labels
    auc_plt.xlabel('False Positive Rate')
    auc_plt.ylabel('True Positive Rate')
    # show the legend
    auc_plt.legend()
    # show the plot
    auc_fig = auc_plt.gcf()
    auc_plt.show()
    auc_plt.draw()
    auc_fig.savefig('plots/auc_plt.png')
    auc_plt.clf()

    # precision, recall, thresholds = precision_recall_curve(test_OUTPUT, prediction_probs)
    # plot_precision_recall_vs_threshold(precision, recall, thresholds)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    auc_plt.figure(figsize=(8, 8))
    auc_plt.title("Precision and Recall Scores as a function of the decision threshold")
    auc_plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    auc_plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    auc_plt.ylabel("Score")
    auc_plt.xlabel("Decision Threshold")
    auc_plt.legend(loc='best')
    auc_plt.show()
