from sklearn import metrics
import matplotlib.pyplot as auc_plt
from math import sqrt
import numpy as np
import pandas as pd
import pylab as pl


def plot_AUC(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, threshold, ngram_min, ngram_max):

    fig, ax = auc_plt.subplots()

    # true positive rate
    recall = metrics.recall_score(test_OUTPUT, predicted_OUTPUT)

    precision = metrics.precision_score(test_OUTPUT, predicted_OUTPUT)

    accuracy = metrics.accuracy_score(test_OUTPUT, predicted_OUTPUT)
    interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / len(test_OUTPUT))

    # no skill prediction
    no_skill_probs = [0 for _ in range(len(test_OUTPUT))]

    # calculate AUC score
    model_auc = metrics.roc_auc_score(test_OUTPUT, prediction_probs)
    print('auc: ', model_auc)

    # calculate roc curves
    no_skills_false_pos_rate, no_skill_true_pos_rate, no_skill_thresholds = metrics.roc_curve(test_OUTPUT,
                                                                                              no_skill_probs)
    model_false_positive_rate, model_true_pos_rate, model_thresholds = metrics.roc_curve(test_OUTPUT,
                                                                                         prediction_probs)
    # string to be outputted in a text box
    experiment_params = '\n'.join((
        'Balancing type:  {}'.format(balancing_type),
        'Solver:  {}'.format(solver),
        'Threshold:  {}'.format(threshold),
        'N-grams:  ({}, {})'.format(ngram_min, ngram_max),
        'AUC:  {} (CI {} - {})'.format(round(model_auc, 3),
                                       round(model_auc - interval/2, 3),
                                       round(model_auc + interval/2, 3)),
        'Recall:  %.3f' % (recall,),
        'Precision:  %.3f' % (precision,),
    ))

    # plot the auc curves
    ax.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='Logistic')
    ax.plot(no_skills_false_pos_rate, no_skill_true_pos_rate, linestyle='--', label='No skills')
    props = dict(boxstyle='round', facecolor='none', edgecolor='grey', alpha=0.7)

    # place a text box in the bottom right
    x, y = get_text_coordinates(balancing_type)
    ax.text(x, y, experiment_params, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # axis labels
    auc_plt.xlabel('False Positive Rate')
    auc_plt.ylabel('True Positive Rate')
    auc_plt.title('Area Under the ROC Curve')
    auc_plt.legend(bbox_to_anchor=(0.97, 0.38), loc='lower right', edgecolor='grey')

    plt_fig = auc_plt.gcf()
    plt_fig.tight_layout()
    plt_fig.savefig('plots/auc_{}_{}_{}_ngram_{}_{}.png'.format(balancing_type, solver, threshold,
                                                 ngram_min, ngram_max))


def get_text_coordinates(balancing_type):

    if balancing_type == 'SMOTE':
        return 0.61, 0.36
    elif balancing_type == 'over-sample positives':
        return 0.465, 0.36
    else:
        return 0.465, 0.36


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
