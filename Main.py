import csv
import re
import pandas as pd
from smoke_extract import *
import numpy as np

data = {}


def read_file(filedir):

    global data
    # read data into pandas dataframe
    data = pd.read_csv(filedir, delimiter=',', low_memory=False, nrows=1000)


read_file('../NOTEEVENTS.csv')
data = data[['SUBJECT_ID', 'TEXT']]
extracted_data = pd.DataFrame(columns=['SUBJECT_ID', 'cigarettes_per_day', 'packs_per_year',
                                       'years_of_smoking', 'time_since_quitting', 'age_of_quitting'])

for i in data['TEXT']:
    if find_smoke(i):
        print("SMOKER")


