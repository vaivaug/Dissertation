"""
Main file to run the code. Two options: enter parameters using Gui or set parameters manually
"""
from Gui import run_gui
from end_to_end import get_data_train_predict, plot_evaluation_metrics
import numpy as np


# variable to know if parameters are entered from Gui or manually
parameters_set_manually = True
# files to store the experiment results
sub_sample_neg_results_file = 'experiments/experiment_result_images/sub_sample_neg_results.csv'
over_sample_pos_results_file = 'experiments/experiment_result_images/over_sample_pos_results.csv'
smote_results_file = 'experiments/experiment_result_images/smote_results.csv'


if parameters_set_manually:

    threshold = 0.6
    balancing_types = ['SMOTE', 'sub-sample negatives', 'over-sample positives']
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ngram_min = 1
    ngram_max = 1
    test_OUTPUT, predicted_OUTPUT, prediction_probs = get_data_train_predict(threshold, balancing_types[1],
                                                                             solvers[0], ngram_min, ngram_max)
    # plot confusion_matrix, AUC, print accuracy
    plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs)

else:
    # enter parameters from Gui
    run_gui()

