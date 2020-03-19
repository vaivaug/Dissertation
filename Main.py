"""
Main file to run the code. Two options: enter parameters using Gui or set parameters manually
"""
from Gui import run_gui
from end_to_end import run_end_to_end
import numpy as np


# variable to know if parameters are entered from Gui or manually
parameters_set_manually = True
# files to store the experiment results
sub_sample_neg_results_file = 'experiments/experiment_result_files/sub_sample_neg_results.csv'
over_sample_pos_results_file = 'experiments/experiment_result_files/over_sample_pos_results.csv'
smote_results_file = 'experiments/experiment_result_files/smote_results.csv'


if parameters_set_manually:

    threshold = 0.4
    balancing_types = ['SMOTE', 'sub-sample negatives', 'over-sample positives']
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ngram_min = 1
    ngram_max = 1
    run_end_to_end(threshold, balancing_types[1], solvers[0], ngram_min, ngram_max)

else:
    # enter parameters from Gui
    run_gui()

