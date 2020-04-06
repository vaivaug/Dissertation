"""
Main file to run the code. Two options: enter parameters using Gui or set parameters manually
"""
from Gui import run_gui
from program_run_start_to_end import predict_test_validation_set, plot_evaluation_metrics, predict_cross_val_train_set

from evaluation.confusion_matrix import get_confusion_matrix
from models.LogisticRegression import plot_word_importance

balancing_types = ["SMOTE", "sub-sample negatives", "over-sample positives"]
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# variable to know if parameters are entered from Gui or manually selected in the code
set_parameters_gui = True

if not set_parameters_gui:

    threshold = 0.4
    balancing_type = balancing_types[1]
    solver = solvers[0]
    ngram_min = 1
    ngram_max = 2
    test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_test_validation_set(threshold, balancing_type,
                                                                             solver, ngram_min, ngram_max)
    # plot confusion_matrix, AUC, print accuracy
    plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs,
                            balancing_type, solver, threshold, ngram_min, ngram_max)

else:
    # enter parameters from Gui
    run_gui()

