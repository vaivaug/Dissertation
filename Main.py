
from Gui import run_gui
from end_to_end import run_end_to_end

# variable to know if parameters are entered from Gui or manually
parameters_set_manually = False

if parameters_set_manually:

    threshold = 0.3
    balancing_type = "over-sample positives"
    solver ='lbfgs'
    ngram_min = 1
    ngram_max = 1
    run_end_to_end(threshold, balancing_type, solver, 1, 2)

else:
    # enter parameters from Gui
    run_gui()

