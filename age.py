import numpy as np
import re, datetime
from dateutil.relativedelta import relativedelta


def get_data_with_age_column(data):

    data['AGE'] = np.nan

    for index, row in data.iterrows():
        print(row['TEXT'])
        age = get_age_from_date_of_birth(row['TEXT'])
        if age is not None:
            data.set_value(index, 'AGE', age)

    return data

'''
1. search for Date of Birth, Discharge date minus d of b
2. else search for 'year' 
3.  search for age'''


def get_age_from_date_of_birth(text):

    word = r"\W*([\w]+)"
    date_of_birth_string = re.search(r'\W*{}{}'.format("Date of Birth", word * 4), text, re.IGNORECASE)

    # date of birth contained inside text
    if date_of_birth_string is not None:

        # get discharge date and subtract date of birth to get age
        discharge_date_string = re.search(r'\W*{}{}'.format("Discharge Date", word * 4), text, re.IGNORECASE)

        if discharge_date_string is not None:

            date_of_birth = re.search(r'\d{4}-\d{1,2}-\d{1,2}', date_of_birth_string.group())
            discharge_date = re.search(r'\d{4}-\d{1,2}-\d{1,2}', discharge_date_string.group())
            if date_of_birth is not None and discharge_date is not None:
                age = relativedelta(datetime.datetime.strptime(discharge_date.group(), '%Y-%m-%d').date(),
                                    datetime.datetime.strptime(date_of_birth.group(), '%Y-%m-%d').date()).years
                if age >= 0:
                    return age

    return None


def search(text, n):
    '''Searches for text, and retrieves n words either side of the text, which are retuned seperatly'''
    word = r"\W*([\w]+)"
    print(type(text))
    print(type('hahdasjdnskadsa dsakdlsa'))
    age = re.search(r'{}\W*{}{}'.format(word * n, "year", word * n), text)
    if age is None:
        age = re.search(r'{}\W*{}{}'.format(word * n, "age", word * n), text)

    print(age)
