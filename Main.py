
from Gui import run_gui
from end_to_end import run_end_to_end

# variable to know if parameters are entered from Gui or manually
parameters_set_manually = False
sub_sample_neg_resut_file = 'experiment_results/sub_sample_neg_results.csv'

if parameters_set_manually:


    threshold = 0.3
    balancing_types = ['SMOTE', 'sub-sample negatives', 'over-sample positives']
    solvers =['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    ngram_min = 1
    ngram_max = 1
    run_end_to_end(threshold, balancing_types[0], solvers[0], 1, 2)

else:
    # enter parameters from Gui
    run_gui()

