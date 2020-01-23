import numpy as np
import re, datetime


def get_data_with_age_column(data):

    data['AGE'] = np.nan

    for index, row in data.iterrows():
        print(row['TEXT'])
        get_age_date_of_birth(row['TEXT'])

    return data

'''
1. search for Date of Birth, Discharge date minus d of b
2. else search for 'year' 
3.  search for age'''


def get_age_date_of_birth(text):

    word = r"\W*([\w]+)"
    date_of_birth_string = re.search(r'\W*{}{}'.format("Date of Birth", word * 4), text, re.IGNORECASE)

    # date of birth contained inside text
    if date_of_birth_string is not None:

        date_of_birth_string = date_of_birth_string.group(0)

        # get discharge date and subtract date of birth to get age
        discharge_date_string = re.search(r'\W*{}{}'.format("Discharge Date", word * 4), text, re.IGNORECASE).group(0)

        date_of_birth = re.search(r'\d{4}-\d{1,2}-\d{1,2}', date_of_birth_string)
        '''if date_of_birth is None:
            date_of_birth = re.search(r'\d{4}-\d{1}-\d{2}', date_of_birth_string)
        if date_of_birth is None:
            date_of_birth = re.search(r'\d{4}-\d{1}-\d{1}', date_of_birth_string)
        if date_of_birth is None:
            date_of_birth = re.search(r'\d{4}-\d{2}-\d{1}', date_of_birth_string)'''


        discharge_date = re.search(r'\d{4}-\d{1,2}-\d{1,2}', discharge_date_string)
        '''if discharge_date is None:
            discharge_date = re.search(r'\d{4}-\d{1}-\d{2}', discharge_date_string)
        if discharge_date is None:
            discharge_date = re.search(r'\d{4}-\d{1}-\d{1}', discharge_date_string)
        if discharge_date is None:
            discharge_date = re.search(r'\d{4}-\d{2}-\d{1}', discharge_date_string)'''

        print(date_of_birth)
        print(discharge_date)




def search(text, n):
    '''Searches for text, and retrieves n words either side of the text, which are retuned seperatly'''
    word = r"\W*([\w]+)"
    print(type(text))
    print(type('hahdasjdnskadsa dsakdlsa'))
    age = re.search(r'{}\W*{}{}'.format(word * n, "year", word * n), text)
    if age is None:
        age = re.search(r'{}\W*{}{}'.format(word * n, "age", word * n), text)

    print(age)
