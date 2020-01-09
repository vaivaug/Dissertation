
import pandas as pd
import csv
from read_data import *

# from textblob import TextBlob
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

filedir_notes = '../NOTEEVENTS.csv'
filedir_notes_subset = '../NOTEEVENTS_SUBSET.csv'
filedir_admissions = '../ADMISSIONS.csv'

df_notes = read_file(filedir_notes)
df_admissions = read_file(filedir_admissions)

diagnosis_contains_cancer = df_admissions[df_admissions['DIAGNOSIS'].str.contains('LUNG CA', na=False)]
print(diagnosis_contains_cancer.count())

df_notes_dis_sum = df_notes.loc[df_notes.CATEGORY == 'Discharge summary']
print('Discharge summary', len(df_notes_dis_sum))

result = get_df_merged(df_notes, df_admissions)
result.head(100).to_csv('../MERGED.csv')

print(result)
print(len(result))


'''
print(len(df_notes.groupby(['SUBJECT_ID'], as_index=False)['TEXT'].sum()))
print(len(df_notes.groupby(['HADM_ID'], as_index=False)['TEXT'].sum()))
print('admissions')
print(len(df_admissions))
print(len(df_admissions.groupby(['HADM_ID'], as_index=False).sum()))
'''


'''
patients_counter = 0
for i in data['SUBJECT_ID']:
    patients_counter += 1
    print(i)

print("patients counter out of 1000 rows: ", patients_counter)
'''