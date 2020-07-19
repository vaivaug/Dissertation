"""
Contains functions to read the data, select 'discharge summaries', merge tables, add add the Output column
"""
import numpy as np
import pandas as pd
from data_preparation.age_from_table import get_data_with_age_column

filedir_notes = '../NOTEEVENTS.csv'
filedir_adm = '../ADMISSIONS.csv'


def get_clean_dataframe():
    """ Clean dataframe to contain only the information we are interested in

    @return: pandas dataframe, clean input data
    """
    adm = get_adm_dataframe()
    notes = get_notes_dataframe()

    notes_adm = get_merged_dataframe(notes, adm)

    notes_adm = get_dataframe_with_outputs(notes_adm)
    notes_adm = get_dataframe_no_newborn(notes_adm)
    notes_adm = get_clean_TEXT_column(notes_adm)
    print('here1')
    print(notes_adm)
    notes_adm = get_notes_with_age(notes_adm)

    print('here2')
    print(notes_adm)
    print('before cancer remove: ', notes_adm['OUTPUT'].value_counts())
    notes_adm = remove_cancer_rows(notes_adm)
    print('here3')
    print(notes_adm)
    print('after cancer remove: ', notes_adm['OUTPUT'].value_counts())

    return notes_adm


def get_adm_dataframe():
    """ Read admissions table

    @return: pandas dataframe containing data from admissions table
    """
    adm = pd.read_csv(filedir_adm)
    adm = adm.sort_values(['SUBJECT_ID'])
    adm = adm.reset_index(drop=True)
    return adm


def get_notes_dataframe():
    """ Read noteevents table, select discharge summaries. Join summaries if multiple exist

    @return: pandas dataframe containing data from noteevents table
    """

    notes = pd.read_csv(filedir_notes)
    # select only the discharge summary column
    # notes = notes.loc[notes.CATEGORY == 'Discharge summary']

    # join the discharge summaries where multiple exist, pick minimum chartdate
    notes_dis_sum_final = (notes.groupby(['SUBJECT_ID', 'HADM_ID'])).aggregate(
        {'CHARTDATE': 'min', 'TEXT': 'sum'}).reset_index()

    # check there is only one discharge summary per person
    assert notes_dis_sum_final.duplicated(['HADM_ID']).sum() == 0, 'Multiple discharge summaries per admission'
    return notes_dis_sum_final


def remove_cancer_rows(notes):
    """Lung cancer patients are identified. Make OUTPUT value 1 for all the discharge summaries
    for lung cancer patients. Then remove rows of lung cancer patients that are after the lung cancer diagnosis
    (including the lung cancer diagnosis row)

    @return:
    """
    subject_ids_checked = []
    indexes_to_drop = []
    count_rows_removed = 0
    lung_cancer_words = ['lung cancer', 'carcinoma', 'cancer', 'lung tumor', 'lung ca', 'small cell ca']

    for cur_index, row in notes.iterrows():
        subject_ID = row['SUBJECT_ID']

        if row['OUTPUT'] == 1 and subject_ID not in subject_ids_checked:

            # find all rows with the same subject_ID, update OUTPUT values for those rows
            notes.loc[notes['SUBJECT_ID'] == subject_ID, 'OUTPUT'] = 1
            indexes = notes.index[notes['SUBJECT_ID'] == subject_ID].tolist()

            for i in indexes:
                if notes.iloc[i]['CHARTDATE'] >= notes.iloc[cur_index]['CHARTDATE']:
                    indexes_to_drop.append(i)

            subject_ids_checked.append(subject_ID)

    notes.drop(indexes_to_drop, inplace=True)

    notes = (notes.groupby(['SUBJECT_ID'])).agg({'HADM_ID': lambda col: col.tolist(),
                                                 'DIAGNOSIS': ','.join, 'TEXT': ' '.join,
                                                 'AGE': 'max', 'OUTPUT': 'max'}).reset_index()

    return notes


def get_merged_dataframe(notes, adm):
    """ Only keep HADM_ID, DIAGNOSIS, TEXT columns, use left merge

    @param notes: pandas dataframe for noteevents table
    @param adm: pandas dataframe for admissions table
    @return: merged pandas dataframe
    """
    print('before merge')
    print('adm')
    print(adm)
    print('notes')
    print(notes)
    notes_adm = pd.merge(adm[['HADM_ID', 'SUBJECT_ID', 'DIAGNOSIS']],
                         notes[['HADM_ID', 'CHARTDATE', 'TEXT']],
                         on=['HADM_ID'],
                         how='left')
    return notes_adm


def get_dataframe_with_outputs(notes_adm):
    """ Lung Cancer patients are given value 1, OUTPUT has value 0 otherwise

    @param notes_adm: merged pandas dataframe with HADM_ID, DIAGNOSIS, TEXT columns
    @return: pandas dataframe with OUTPUT column and values
    """

    notes_adm['OUTPUT'] = (notes_adm.DIAGNOSIS.str.contains('(LUNG CA)|MESOTHELIOMA|(LUNG NEOPLASM)|\
                           (SMALL CELL LUNG CA)|(LOBE CA)|(MALIGNANT PLEURAL EFFUSION)', na=False) |
                           (notes_adm.DIAGNOSIS.str.contains('LUNG', na=False) &
                           notes_adm.DIAGNOSIS.str.contains('TUMOR', na=False)) |
                           (notes_adm.DIAGNOSIS.str.contains('LUNG', na=False) &
                            notes_adm.DIAGNOSIS.str.contains('CANCER', na=False)) |
                           (notes_adm.DIAGNOSIS.str.contains('SMALL CELL', na=False) &
                            notes_adm.DIAGNOSIS.str.contains('CANCER', na=False)) |
                           (notes_adm.DIAGNOSIS.str.contains('SMALL CELL', na=False) &
                            notes_adm.DIAGNOSIS.str.contains('CARCINOMA', na=False))
                           ).astype('int')

    return notes_adm


def get_dataframe_no_newborn(notes_adm):
    """ Patients with NEWBORN Diagnosis are removed

    @param notes_adm: merged pandas dataframe with HADM_ID, DIAGNOSIS, TEXT, OUTPUT columns
    @return: pandas dataframe with no NEWBORN
    """
    notes_adm_final = notes_adm[notes_adm.DIAGNOSIS.str.contains('NEWBORN')==False]

    return notes_adm_final


def get_clean_TEXT_column(notes_adm):
    """ Clean TEXT column values, remove empty discharge summaries

    @param notes_adm: pandas dataframe with HADM_ID, DIAGNOSIS, TEXT, OUTPUT columns
    @return: pandas dataframe with cleaned TEXT column
    """

    notes_adm.TEXT = notes_adm.TEXT.str.replace('\n', ' ')
    notes_adm.TEXT = notes_adm.TEXT.str.replace('\r', ' ')
    notes_adm['TEXT'] = notes_adm['TEXT'].str.lower()
    notes_adm['TEXT'].replace(' ', np.nan, inplace=True)

    # drop rows with empty discharge summaries
    notes_adm.dropna(subset=['TEXT'], inplace=True)

    return notes_adm


def get_notes_with_age(notes_adm):
    notes_adm = get_data_with_age_column(notes_adm)
    return notes_adm


