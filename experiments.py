from program_run_start_to_end import predict_cross_val_train_set
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# for each balancing type, we run the code with different solvers, different thresholds and different ngrams
def run_experiment_balance_solver(balancing_type, solver):

    thresholds_list = np.arange(0, 1.05, 0.05)
    print(type(thresholds_list))
    print(thresholds_list)
    ngrams_list = [[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]

    for threshold in thresholds_list:
        for ngram in ngrams_list:

            train_OUTPUT, predicted_OUTPUT, prediction_probs = predict_cross_val_train_set(0.4,
                                                                                          balancing_type,
                                                                                          solver,
                                                                                          ngram[0],
                                                                                          ngram[1])
            plot(train_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, threshold, ngram[0],
                                  ngram[1])
            break
        break


def run_all_experiments():

    balancing_types = ['SMOTE', 'sub-sample negatives', 'over-sample positives']
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    # for each pair of balancing type and solver, run experiments with different threshold and ngram values
    for balancing_type in balancing_types:
        for solver in solvers:
            print('Balancing type: ', balancing_type, '   solver:  ', solver)
            run_experiment_balance_solver(balancing_type, solver)


def plot(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, threshold, ngram_min, ngram_max):

    fig, ax = plt.subplots()
    # true positive rate
    recall = metrics.recall_score(test_OUTPUT, predicted_OUTPUT)

    # no skill prediction
    no_skill_probs = [0 for _ in range(len(test_OUTPUT))]

    # calculate scores
    no_skill_auc = metrics.roc_auc_score(test_OUTPUT, no_skill_probs)
    model_auc = metrics.roc_auc_score(test_OUTPUT, prediction_probs)

    # calculate roc curves
    no_skills_false_pos_rate, no_skill_true_pos_rate, no_skill_thresholds = metrics.roc_curve(test_OUTPUT, no_skill_probs)
    model_false_positive_rate, model_true_pos_rate, model_thresholds = metrics.roc_curve(test_OUTPUT, prediction_probs)

    experiment_params = '\n'.join((
        'Balancing type:  {}'.format(balancing_type),
        'Solver:  {}'.format(solver),
        'Threshold:  {}'.format(threshold),
        'N-grams:  ({}, {})'.format(ngram_min, ngram_max),
        'AUC:  %.3f' % (model_auc,),
        'Recall:  %.3f' % (recall,),
    ))

    ax.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='Logistic')
    ax.plot(no_skills_false_pos_rate, no_skill_true_pos_rate, linestyle='--', label='No skills')
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='none', edgecolor='black', alpha=0.7)

    # place a text box in upper left in axes coords
    ax.text(0.41, 0.3, experiment_params, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')

    plt.show()


def plot_evaluation_image(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, ngram_min, ngram_max):

    # true positive rate
    recall = metrics.recall_score(test_OUTPUT, predicted_OUTPUT)

    model_auc = metrics.roc_auc_score(test_OUTPUT, prediction_probs)
    print('Logistic: ROC AUC=%.3f' % model_auc)

    # calculate roc curves
    model_false_positive_rate, model_true_pos_rate, model_thresholds = metrics.roc_curve(test_OUTPUT, prediction_probs)

    # plot the roc curve for the model
    plt.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='Logistic Regression')
    plt.title('Area Under the ROC Curve')

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }

   # plt.text(6, 6, 'some text\n another line', fontdict=font)

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # show the legend
    plt.legend(title='Recall: {}\nAUC={}' .format(recall, round(model_auc, 3)), loc='lower right')
    # Tweak spacing to prevent clipping of ylabel
   #  plt.subplots_adjust(left=0.15)
    plt.show()

    '''
    plt.tight_layout()
    # show the plot
    auc_fig = plt.gcf()
    auc_fig.tight_layout()
    plt.show()
    plt.draw()
    auc_fig.savefig('plots/auc_plt.png')
    plt.clf()
    '''


# run_all_experiments()
run_experiment_balance_solver('sub-sample negatives', 'lbfgs')
