from tkinter import *

from tkinter.ttk import *

window = Tk()
window.title("Set up program parameters")
window.geometry('600x400')


# Threshold
threshold_label = Label(window, text="  Threshold: ")
threshold_label.grid(column=0, row=0)
txt = Entry(window,width=12)
txt.grid(column=1, row=0)

def clicked():

    threshold = txt.get()
    # lbl.configure(text= res)

# btn = Button(window, text="Click Me", command=clicked)

# btn.grid(column=2, row=0)


var1 = StringVar()
var2 = StringVar()

var1.set(0)
var2.set(0)


data_balancing_label = Label(window, text="  Train data balancing type: ")
data_balancing_label.grid(column=0, row=1)

data_balance_type1 = Radiobutton(window,text='SMOTE', variable=var1, value=1)
data_balance_type2 = Radiobutton(window,text='sub-sample negatives', variable=var1, value=2)
data_balance_type3 = Radiobutton(window,text='over-sample positives', variable=var1, value=3)

data_balance_type1.grid(column=0, row=2)
data_balance_type2.grid(column=1, row=2)
data_balance_type3.grid(column=2, row=2)

rad1 = Radiobutton(window,text='1', variable=var2, value=4)
rad2 = Radiobutton(window,text='2', variable=var2, value=5)
rad3 = Radiobutton(window,text='3', variable=var2, value=6)

rad1.grid(column=0, row=3)
rad2.grid(column=1, row=3)
rad3.grid(column=2, row=3)

window.mainloop()
