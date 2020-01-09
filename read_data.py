
import pandas as pd


def read_file(filedir, number_of_rows):

    data = pd.read_csv(filedir, delimiter=',', low_memory=False, nrows=number_of_rows)
    return data


def read_file(filedir):

    data = pd.read_csv(filedir, delimiter=',', low_memory=False)
    return data


def get_df_merged(df_notes, df_admissions):

    df_notes = df_notes.groupby(['HADM_ID'], as_index=False).agg({'SUBJECT_ID': 'sum', 'TEXT': 'sum'})
    print(len(df_notes))

    result = pd.merge(df_notes[['SUBJECT_ID', 'HADM_ID', 'TEXT']], df_admissions[['HADM_ID', 'DIAGNOSIS']],
                      on='HADM_ID', how='inner')
    return result