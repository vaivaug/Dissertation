"""
Main file to run the code. Two options: enter parameters using Gui or set parameters manually
"""
from Gui import run_gui
from program_run_start_to_end import predict_test_validation_set, plot_evaluation_metrics, predict_cross_val_train_set
import numpy as np
import datetime

balancing_types = ["SMOTE", "sub-sample negatives", "over-sample positives"]
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
thresholds = [0.1, 0.2, 0.4, 0.6, 0.8]
ngrams = [[1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]]

# variable to know if parameters are entered from Gui or manually selected in the code
set_parameters_gui = False
start = datetime.datetime.now()

if not set_parameters_gui:

    threshold = 0.05
    balancing_type = "over-sample positives"
    solver = 'liblinear'
    ngram_min = 1
    ngram_max = 3

    test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_test_validation_set(threshold, balancing_type,
                                                                                  solver, ngram_min, ngram_max)

    # plot confusion_matrix, AUC, print accuracy
    for thr in thresholds:
        predicted_OUTPUT = np.where(prediction_probs > thr, 1, 0)
        plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs,
                                    balancing_type, solver, thr, ngram_min, ngram_max)
    end = datetime.datetime.now()
    print('time taken: ', end - start)

else:
    # enter parameters from Gui
    run_gui()

