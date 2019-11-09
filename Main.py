
import pandas as pd
import csv
from split_data_pos_neg import initialise_file_columns



from word_analysis_functions import *
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


data = {}


def read_file(filedir):

    global data
    # read data into pandas dataframe
    data = pd.read_csv(filedir, delimiter=',', low_memory=False,
                       nrows=1000)


initialise_file_columns('positive.csv')

'''
with open('negative.csv', 'a') as file_pos:
    file_pos.write("TEXT,")
'''


def append_pos_lung_cancer(text):
    with open('positive.csv', 'a') as file_pos:
        file_pos_writer = csv.writer(file_pos, delimiter=',')
        file_pos_writer.writerow([text, '1'])

'''
def append_neg_lung_cancer(text):
    with open('negative.csv', 'a') as file_neg:
        file_neg.write(text)

'''

read_file('../NOTEEVENTS.csv')
# data = data[['SUBJECT_ID', 'TEXT']]

all_text = ""
counter_lung_cancer = 0
counter_no_lung_cancer = 0

for text in data['TEXT']:
    if "lung cancer" in text:
        counter_lung_cancer += 1
        append_pos_lung_cancer(text)
    else:
        counter_no_lung_cancer += 1
        # append_neg_lung_cancer(text)
    all_text += text


positive_df = pd.read_csv('positive.csv')

print("lung cancer contained in: ", counter_lung_cancer)


print("positive rows: ", positive_df.shape[0])

# print("lung cancer not contained in: ", counter_no_lung_cancer, "   Number of rows in negative: ",
#      sum(1 for row in csv.reader('negative.csv')))

data = data.groupby(['SUBJECT_ID'], as_index=False)




'''
patients_counter = 0
for i in data['SUBJECT_ID']:
    patients_counter += 1
    print(i)

print("patients counter out of 1000 rows: ", patients_counter)
'''



# frequency_distribution(all_text)
# longer_words_appear_the_most(all_text, 7, 100)
# get_bigrams(all_text)
