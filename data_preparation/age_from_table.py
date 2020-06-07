"""
Contains a function which adds an AGE column to all the patients in the dataframe.
If the age is not extracted, column value is NaN
Age filter was not used at the end
"""
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.0f}'.format
import re
import datetime
from dateutil.relativedelta import relativedelta

filedir_patients = '../PATIENTS.csv'


def get_data_with_age_column(data):
    """ Create Age column for the given dataset. Age is mentioned in free text, no consistent structure.
    The following code extracts the age of all the patients apart from around 200. The code is not used in the final
    program run.

    @param data: pandas dataframe, containing rows of training or testing data sets
    @return: pandas dataframe, containing rows of training or testing data sets, added AGE column
    """

    # get SUBJECT_ID and DOB dataframe
    patient_dob = get_dob_from_patients_table()
    data = data.reset_index(drop=True)

    # merge data and patient_dob tables by SUBJECT_ID
    data = pd.merge(patient_dob[['SUBJECT_ID', 'DOB']],
                         data[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'DIAGNOSIS', 'TEXT', 'OUTPUT']],
                         on=['SUBJECT_ID'],
                         how='right')

    # find age from DOB and CHARTDATE
    now = pd.Timestamp('now')
    data['CHARTDATE'] = pd.to_datetime(data['CHARTDATE'], format='%Y-%m-%d %H:%M:%S')
    data['CHARTDATE'] = data['CHARTDATE'].where(data['CHARTDATE'] < now, data['CHARTDATE'] - np.timedelta64(100, 'Y'))

    data['DOB'] = pd.to_datetime(data['DOB'], format='%Y-%m-%d %H:%M:%S')
    data['DOB'] = data['DOB'].where(data['DOB'] < now, data['DOB'] - np.timedelta64(100, 'Y'))

    data['AGE'] = (data['CHARTDATE'] - data['DOB']).astype('<m8[Y]')
    data = data.drop(columns=['CHARTDATE', 'DOB'])

    return data


def get_dob_from_patients_table():
    patient_dob = pd.read_csv(filedir_patients)
    patient_dob = patient_dob[['SUBJECT_ID', 'DOB']]
    patient_dob = patient_dob.reset_index(drop=True)
    return patient_dob

