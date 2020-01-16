'''
Contains functions to read the data, select 'dischange summaries', merge tables, add Output column
'''

import pandas as pd

filedir_notes = '../NOTEEVENTS.csv'
filedir_adm = '../ADMISSIONS.csv'


def get_clean_dataframe():
    adm = get_adm_dataframe()
    notes = get_notes_dataframe()

    # check there is only one discharge summary per person. Can delete this row later
    assert notes.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'
    notes_adm = get_merged_dataframe(notes, adm)
    notes_adm = get_dataframe_with_outputs(notes_adm)
    notes_adm = get_dataframe_no_newborn(notes_adm, adm)
    return notes_adm


def get_adm_dataframe():
    # read admissions table
    adm = pd.read_csv(filedir_adm)
    adm = adm.sort_values(['SUBJECT_ID'])
    adm = adm.reset_index(drop=True)
    return adm


def get_notes_dataframe():
    # read noteevenets table
    notes = pd.read_csv(filedir_notes)
    # select only the discharge summary column
    notes_dis_sum = notes.loc[notes.CATEGORY == 'Discharge summary']
    # select the last 'discharge summary'.
    '''TODO: try joining the discharge summaries where multiple exist'''

    notes_dis_sum = (notes_dis_sum.groupby(['SUBJECT_ID', 'HADM_ID'])).aggregate({'TEXT': 'sum'}).reset_index()

    assert notes_dis_sum.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'
    return notes_dis_sum


'''
# Select duplicate rows except first occurrence based on all columns
duplicated = notes_dis_sum[notes_dis_sum.duplicated(subset=['HADM_ID'], keep=False)]
duplicated.to_csv('DUPLICATED.csv')

print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicated)
'''


def get_merged_dataframe(notes, adm):

    notes_adm = pd.merge(adm[['HADM_ID','DIAGNOSIS']],
                            notes[['HADM_ID','TEXT']],
                            on=['HADM_ID'],
                            how='left')
    return notes_adm


def get_dataframe_with_outputs(notes_adm):
    '''TODO: can be improved by adding more keywords meaning lung cancer.'''
    notes_adm['OUTPUT'] = (notes_adm.DIAGNOSIS.str.contains('LUNG CA') |
                                (notes_adm.DIAGNOSIS.str.contains('LUNG', na=False) &
                                notes_adm.DIAGNOSIS.str.contains('TUMOR', na=False)) |
                                (notes_adm.DIAGNOSIS.str.contains('LUNG', na=False) &
                                notes_adm.DIAGNOSIS.str.contains('CANCER', na=False))
                                ).astype('int')
    return notes_adm


def get_dataframe_no_newborn(notes_adm, adm):
    # remove newborn
    notes_adm_final = notes_adm[notes_adm.DIAGNOSIS.str.contains('NEWBORN')==False]
    print(notes_adm_final)
    print(notes_adm_final.groupby('OUTPUT').count())
    # check number of rows is the same. Can delete later
    assert len(adm) == len(notes_adm), 'Number of rows is higher'
    return notes_adm_final

