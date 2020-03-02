from tkinter import *

from tkinter.ttk import *

global threshold, balancing_type, model, ngram_start, ngram_end

from Main import run_main


def run_gui():

    global threshold, balancing_type, model, ngram_start, ngram_end

    window = Tk()
    window.title("Set up program parameters")
    window.configure(background='peach puff')
    window.geometry('800x600')


    # threshold label
    threshold_label = Label(window, text="  Threshold:")
    threshold_label.config(font=("Courier bold", 12), background='peach puff')
    threshold_label.grid(column=0, row=0, sticky=W, padx=10, pady=10)

    # enter threshold window
    threshold_value = Entry(window, width=12)
    threshold_value.grid(column=1, row=0, sticky=W, padx=10, pady=10)


    # label
    data_balancing_label = Label(window, text="  Train data balancing type: ")
    data_balancing_label.config(font=("Courier bold", 12), background='peach puff')
    data_balancing_label.grid(column=0, row=2, sticky=W, padx=10, pady=10)
    data_balancing_type = Combobox(window, state='readonly')
    data_balancing_type['values']= ("SMOTE", "sub-sample negatives", "over-sample positives")
    data_balancing_type.config(font=("Courier bold", 10))
    data_balancing_type.current(0)
    data_balancing_type.grid(column=1, row=2, padx=10, pady=10)

    # label
    model_label = Label(window, text="  Model: ")
    model_label.config(font=("Courier bold", 12), background='peach puff')
    model_label.grid(column=0, row=4, sticky=W, padx=10, pady=10)

    model_type = Combobox(window, state='readonly')
    model_type['values']= ("Logistic Regression", "SVM")
    model_type.config(font=("Courier bold", 10))
    model_type.current(0)

    model_type.grid(column=1, row=4, padx=10, pady=10)


    def clicked():

        threshold = float(threshold_value.get())
        balancing_type = data_balancing_type.get()
        model = model_type.get()
        ngram_start = int(ngram_min.get())
        ngram_end = int(ngram_max.get())
        print(model, '   ', balancing_type, '  ', threshold, '  ', ngram_start, '  ', ngram_end)

        run_main(threshold, balancing_type, model, ngram_start, ngram_end)


    # ngram
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

    run_button = Button(window, text="Run Model", command=clicked)
    run_button.grid(column=1, row=6, padx=10, pady=10)

    window.mainloop()

run_gui()