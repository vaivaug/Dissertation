"""
Gui window to enter the parameters. Once the 'Run Code' button is clicked, the entered parameters are
used to run the end to end process: data cleaning, training the model, making predictions
and plotting the prediction results
"""
from tkinter import *
from tkinter.ttk import *
global threshold, balancing_type, solver, ngram_start, ngram_end
from end_to_end import get_data_train_predict, plot_evaluation_metrics
#from PIL import ImageTk, Image

global threshold, data_balancing_type, solver, ngram_start, ngram_end


def run_gui():

    window = Tk()
    window.title("Program Parameters and Model Results")
    window.configure(background='peach puff')
    window.geometry('800x600')

    create_threshold_param(window)
    create_data_balance_param(window)
    create_solver_selection(window)
    create_ngram_selection(window)

    run_button = Button(window, text="Run Model", command=clicked)
    run_button.grid(column=1, row=6, padx=10, pady=10)

    window.mainloop()


def clicked():

    test_OUTPUT, predicted_OUTPUT, prediction_probs = get_data_train_predict(float(threshold.get()),
                                                                             data_balancing_type.get(),
                                                                             solver.get(),
                                                                             int(ngram_min.get()),
                                                                             int(ngram_max.get()))
    # plot confusion_matrix, AUC, print accuracy
    plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs)

    # show_output_images(window)

'''
def show_output_images(window):
    path = "plots/conf_matrix.jpg"

    # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img = ImageTk.PhotoImage(Image.open(path))

    # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tk.Label(window, image=img)

    # The Pack geometry manager packs widgets in rows or columns.
    panel.pack(side="bottom", fill="both", expand="yes")
'''


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
    solver_label = Label(window, text="  Linear Regression Solver: ")
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
