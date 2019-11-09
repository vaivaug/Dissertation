
import pandas as pd
import csv
from split_data_pos_neg import *


# from textblob import TextBlob
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
data = {}
counter_lung_cancer = 0
counter_no_lung_cancer = 0


def read_file(filedir):

    global data
    data = pd.read_csv(filedir, delimiter=',', low_memory=False, nrows=1000)


def create_training_files():

    initialise_file_columns('positive.csv')
    initialise_file_columns('negative.csv')

    global counter_no_lung_cancer
    global counter_lung_cancer

    for text in data['TEXT']:
        if "lung cancer" in text:
            counter_lung_cancer += 1
            append_file('positive.csv', text, 1)
        else:
            counter_no_lung_cancer += 1
            append_file('negative.csv', text, 0)


read_file('../NOTEEVENTS.csv')

data = data.groupby(['SUBJECT_ID'], as_index=False)['TEXT'].sum()
print(data)

create_training_files()

positive_df = read_file_to_pandas('positive.csv')
negative_df = read_file_to_pandas('negative.csv')

print("lung cancer contained in: ", counter_lung_cancer)

print("positive rows: ", positive_df.shape[0])
print("negative rows: ", negative_df.shape[0])

data = data.groupby(['SUBJECT_ID'], as_index=False)['TEXT'].sum()
print(data)


'''
patients_counter = 0
for i in data['SUBJECT_ID']:
    patients_counter += 1
    print(i)

print("patients counter out of 1000 rows: ", patients_counter)
'''

