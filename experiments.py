from program_run_start_to_end import predict_cross_val_train_set, predict_test_validation_set
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from math import sqrt
from evaluation.confusion_matrix import get_confusion_matrix
import random


# for each balancing type, we run the code with different solvers, different thresholds and different ngrams
def run_experiment_balance_solver(balancing_type, solver):

    thresholds_list = np.arange(0.15, 1.05, 0.05)
    ngrams_list = [[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]

    for threshold in thresholds_list:
        for ngram in ngrams_list:

            test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_cross_val_train_set(round(threshold, 2),
                                                                                          balancing_type,
                                                                                          solver,
                                                                                          ngram[0],
                                                                                          ngram[1])
            plot(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, round(threshold, 2), ngram[0],
                                  ngram[1])


def run_all_experiments():

    balancing_types = ['over-sample positives']
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    # for each pair of balancing type and solver, run experiments with different threshold and ngram values
    for balancing_type in balancing_types:
        for solver in solvers:
            print('Balancing type: ', balancing_type, '   solver:  ', solver)
            run_experiment_balance_solver(balancing_type, solver)


def run_random_experiments_for_balancing_types():

    balancing_types = ['SMOTE', 'sub-sample negatives', 'over-sample positives']
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ngrams_list = [[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]

    # select random solver, random threshold and random ngrams
    # run experiment with those random parameters on each balancing type
    for exp in range(0, 5):
        random_ngram = ngrams_list[random.randint(0, 5)]
        random_solver = solvers[random.randint(0, 4)]
        random_threshold = round(random.uniform(0, 1), 2)

        for balancing_type in balancing_types:
            test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_cross_val_train_set(random_threshold,
                                                                                          balancing_type,
                                                                                          random_solver,
                                                                                          random_ngram[0],
                                                                                          random_ngram[1])
            plot(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, random_solver, random_threshold,
                 random_ngram[0], random_ngram[1])



def plot(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, threshold, ngram_min, ngram_max):

    fig, ax = plt.subplots()

    # true positive rate
    recall = metrics.recall_score(test_OUTPUT, predicted_OUTPUT)

    precision = metrics.precision_score(test_OUTPUT, predicted_OUTPUT)
    print('precision: better is close to 1: ', precision)

    accuracy = metrics.accuracy_score(test_OUTPUT, predicted_OUTPUT)
    print('accuracy: ', accuracy)
    interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / len(test_OUTPUT))
    print('Confidence Interval: %.3f' % interval)

    get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)

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
        'AUC:  {} (CI {} - {})'.format(round(model_auc, 2),
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Area Under the ROC Curve')
    plt.legend(bbox_to_anchor=(0.97, 0.38), loc='lower right', edgecolor='grey')

    plt_fig = plt.gcf()
    plt_fig.tight_layout()
    plt_fig.savefig('experiment_plots/{}-{}-{}-({},{}).png'.format(
       balancing_type, solver, threshold, ngram_min, ngram_max))

    # plt.show()


def get_text_coordinates(balancing_type):

    if balancing_type == 'SMOTE':
        return 0.61, 0.36
    elif balancing_type == 'over-sample positives':
        return 0.465, 0.36
    else:
        return 0.465, 0.36


run_all_experiments()
# run_experiment_balance_solver('SMOTE', 'lbfgs')
# run_random_experiments_for_balancing_types()
