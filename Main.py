"""
Main file to run the code. Two options: enter parameters using Gui or set parameters manually
"""
from Gui import run_gui
from program_run_start_to_end import predict_test_validation_set, plot_evaluation_metrics


# variable to know if parameters are entered from Gui or manually
parameters_set_manually = True

if parameters_set_manually:

    threshold = 0.5
    balancing_types = ['SMOTE', 'sub-sample negatives', 'over-sample positives']
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ngram_min = 1
    ngram_max = 1
    test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_test_validation_set(threshold, balancing_types[1],
                                                                             solvers[1], ngram_min, ngram_max)
    # plot confusion_matrix, AUC, print accuracy
    plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs)

else:
    # enter parameters from Gui
    run_gui()

