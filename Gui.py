"""
Gui window to enter the parameters. Once the 'Run Code' button is clicked, the entered parameters are
used to run the end to end process: data cleaning, training the model, making predictions
and plotting the prediction results
"""
from tkinter import *
from tkinter.ttk import *
global threshold, balancing_type, solver, ngram_start, ngram_end
from program_run_start_to_end import predict_test_validation_set, plot_evaluation_metrics, predict_cross_val_train_set
from PIL import ImageTk, Image

global threshold, data_balancing_type, solver, ngram_start, ngram_end
global window


def run_gui():

    global window

    window = Tk()
    window.title("Program Parameters and Model Results")
    window.configure(background='peach puff')
    window.geometry('800x590')

    create_threshold_param(window)
    create_data_balance_param(window)
    create_solver_selection(window)
    create_ngram_selection(window)

    run_button = Button(window, text="Run Model", command=clicked)
    run_button.grid(column=1, row=6, padx=10, pady=10)

    window.mainloop()


def clicked():

    test_OUTPUT, predicted_OUTPUT, prediction_probs = predict_test_validation_set(float(threshold.get()),
                                                                             data_balancing_type.get(),
                                                                             solver.get(),
                                                                             int(ngram_min.get()),
                                                                             int(ngram_max.get()))
    # plot confusion_matrix, AUC, print accuracy
    plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs, data_balancing_type.get(),
                            solver.get(), float(threshold.get()), int(ngram_min.get()), int(ngram_max.get()))

    show_output_images(window)


def show_output_images(window):

    # explanation of 1 and 0
    conf_matrix_values = Label(window, text="  Predicted values: 1 - at risk of lung cancer, 0 - not at risk of lung cancer")
    conf_matrix_values.config(font=("Courier bold", 12), background='peach puff')
    conf_matrix_values.grid(column=0, row=7, columnspan=3, sticky=W, padx=10, pady=10)

    # AUC
    path = "plots/auc_plt.png"
    image = Image.open(path).resize((360, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.image = photo
    label.grid(column=0, row=8, columnspan=2, padx=11, pady=11)

    # confusion matrix
    path = "plots/conf_matrix_plt.png"
    image = Image.open(path).resize((360, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo)
    label.image = photo
    label.grid(column=2, row=8, columnspan=2, padx=11, pady=11)


def create_threshold_param(window):

    global threshold
    # threshold label
    threshold_label = Label(window, text="  Threshold:")
    threshold_label.config(font=("Courier bold", 12), background='peach puff')
    threshold_label.grid(column=0, row=0, sticky=W, padx=10, pady=10)

    # enter threshold window
    threshold = Entry(window, width=12)
    threshold.grid(column=1, row=0, sticky=W, padx=10, pady=10)


def create_data_balance_param(window):

    global data_balancing_type

    # label
    data_balancing_label = Label(window, text="  Train data balancing type: ")
    data_balancing_label.config(font=("Courier bold", 12), background='peach puff')
    data_balancing_label.grid(column=0, row=2, sticky=W, padx=10, pady=10)

    # drop down selection
    data_balancing_type = Combobox(window, state='readonly')
    data_balancing_type['values'] = ("SMOTE", "sub-sample negatives", "over-sample positives")
    data_balancing_type.config(font=("Courier bold", 10))
    # default value selected
    data_balancing_type.current(0)
    data_balancing_type.grid(column=1, row=2, padx=10, pady=10)


def create_solver_selection(window):

    global solver

    # label
    solver_label = Label(window, text="  Logistic Regression Solver: ")
    solver_label.config(font=("Courier bold", 12), background='peach puff')
    solver_label.grid(column=0, row=4, sticky=W, padx=10, pady=10)

    # drop down selection
    solver = Combobox(window, state='readonly')
    solver['values'] = ("newton-cg", "lbfgs", "liblinear", "sag", "saga")
    solver.config(font=("Courier bold", 10))
    solver.current(0)
    solver.grid(column=1, row=4, padx=10, pady=10)


def create_ngram_selection(window):

    global ngram_min, ngram_max

    # ngram label
    ngram_label = Label(window, text="  Ngram:")
    ngram_label.config(font=("Courier bold", 12), background='peach puff')
    ngram_label.grid(column=0, row=5, sticky=W, padx=10, pady=10)

    # enter ngram window
    ngram_min = Entry(window, width=12)
    ngram_min.grid(column=1, row=5, sticky=W, padx=10, pady=10)

    # enter ngram window
    ngram_max = Entry(window, width=12)
    ngram_max.grid(column=2, row=5, sticky=W, padx=10, pady=10)
