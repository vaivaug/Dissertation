"""
Main file to run the code. Two options: enter parameters using Gui or set parameters manually
"""
from Gui import run_gui
from program_run_start_to_end import predict_test_validation_set, plot_evaluation_metrics, predict_cross_val_train_set

from experiments import plot
from evaluation.confusion_matrix import get_confusion_matrix

# variable to know if parameters are entered from Gui or manually
parameters_set_manually = True

if parameters_set_manually:

    threshold = 0.4
    balancing_types = ['SMOTE', 'sub-sample negatives', 'over-sample positives']
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ngram_min = 1
    ngram_max = 1
    print('on validation set: ')
    test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_test_validation_set(threshold, balancing_types[1],
                                                                             solvers[2], ngram_min, ngram_max)
    # plot confusion_matrix, AUC, print accuracy
    plot(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_types[1], solvers[1], threshold, ngram_min, ngram_max)
    get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)

    print('on train cross validation: ')
    test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_cross_val_train_set(threshold, balancing_types[1],
                                                                                  solvers[2], ngram_min, ngram_max)
    # plot confusion_matrix, AUC, print accuracy
    plot(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_types[1], solvers[1], threshold, ngram_min, ngram_max)
    get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)
else:
    # enter parameters from Gui
    run_gui()

