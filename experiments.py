from program_run_start_to_end import predict_cross_val_train_set
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# for each balancing type, we run the code with different solvers, different thresholds and different ngrams
def run_experiment_balance_solver(balancing_type, solver):

    thresholds_list = [np.arange(0, 1.05, 0.05)]
    print(thresholds_list)
    ngrams_list = [[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]

    for threshold in thresholds_list:
        for ngram in ngrams_list:

            train_OUTPUT, predicted_OUTPUT, prediction_probs = predict_cross_val_train_set(0.4,
                                                                                          balancing_type,
                                                                                          solver,
                                                                                          ngram[0],
                                                                                          ngram[1])
            plot_evaluation_image(train_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, ngram[0],
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


def plot_evaluation_image(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, ngram_min, ngram_max):
    # recall (true positive rate)
    print("Recall:", metrics.recall_score(test_OUTPUT, predicted_OUTPUT))

    model_auc = metrics.roc_auc_score(test_OUTPUT, prediction_probs)
    print('Logistic: ROC AUC=%.3f' % model_auc)

    # calculate roc curves
    model_false_positive_rate, model_true_pos_rate, model_thresholds = metrics.roc_curve(test_OUTPUT, prediction_probs)

    # plot the roc curve for the model
    plt.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='Logistic Regression')

    fig, ax = plt.subplots()
    ax.text(0.05, 0.95, "text here", transform=ax.transAxes, fontsize=14,
        verticalalignment='top')

    plt.show()


    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Area Under the ROC Curve')

    # show the legend
    plt.legend(loc='lower right')


    plt.tight_layout()
    # show the plot
    auc_fig = plt.gcf()
    auc_fig.tight_layout()
    plt.show()
    plt.draw()
    auc_fig.savefig('plots/auc_plt.png')
    plt.clf()


# run_all_experiments()
run_experiment_balance_solver('sub-sample negatives', 'lbfgs')
