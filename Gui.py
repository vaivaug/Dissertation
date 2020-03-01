from tkinter import *

from tkinter.ttk import *

window = Tk()
window.title("Set up program parameters")
window.geometry('600x400')


# threshold label
threshold_label = Label(window, text="  Threshold:")
threshold_label.config(font=("Courier bold", 12))
threshold_label.grid(column=0, row=0, sticky=W, padx=10, pady=10)

# enter threshold window
txt = Entry(window, width=12)
txt.grid(column=1, row=0, sticky=W, padx=10, pady=10)


# label
data_balancing_label = Label(window, text="  Train data balancing type: ")
data_balancing_label.config(font=("Courier bold", 12))
data_balancing_label.grid(column=0, row=2, sticky=W, padx=10, pady=10)
data_balancing_type = Combobox(window)
data_balancing_type['values']= ("SMOTE", "sub-sample negatives", "over-sample positives")
data_balancing_type.current(0)
data_balancing_type.grid(column=1, row=2, padx=10, pady=10)


# label
model_label = Label(window, text="  Model: ")
model_label.config(font=("Courier bold", 12))
model_label.grid(column=0, row=4, sticky=W, padx=10, pady=10)


model_type = Combobox(window)
model_type['values']= ("Logistic Regression", "SVM")
model_type.current(0)

model_type.grid(column=1, row=4, padx=10, pady=10)


def clicked():

    threshold = txt.get()
    balancing_type = data_balancing_type.get()
    model = model_type.get()
    print(model, '   ', balancing_type, '  ', threshold)


run_button = Button(window, text="Run Model", command=clicked)
run_button.grid(column=1, row=5, padx=10, pady=10)


window.mainloop()
