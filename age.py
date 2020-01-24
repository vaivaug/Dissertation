import numpy as np
import re, datetime
from dateutil.relativedelta import relativedelta


def get_data_with_age_column(data):

    # initialise age column with NaN
    data['AGE'] = np.nan

    for index, row in data.iterrows():
        # get age from date of birth
        age = get_age_from_date_of_birth(row['TEXT'])

        if age is None:
            # get age from inside of text
            age = get_age_from_text(" ".join(row['TEXT'].split()[:100]), 3)

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
        end_date_string = re.search(r'\W*{}{}'.format("Discharge Date", word * 4), text, re.IGNORECASE)

        # if discharge date is not available then try Admission date (age should be the same or -1 year)
        if end_date_string is None:
            end_date_string = re.search(r'\W*{}{}'.format("Admission Date", word * 4), text, re.IGNORECASE)

        if end_date_string is not None:

            date_of_birth = re.search(r'\d{4}-\d{1,2}-\d{1,2}', date_of_birth_string.group())
            discharge_date = re.search(r'\d{4}-\d{1,2}-\d{1,2}', end_date_string.group())
            if date_of_birth is not None and discharge_date is not None:
                age = relativedelta(datetime.datetime.strptime(discharge_date.group(), '%Y-%m-%d').date(),
                                    datetime.datetime.strptime(date_of_birth.group(), '%Y-%m-%d').date()).years
                if age >= 0:
                    return age



    return None


def get_age_from_text(text, n):
    '''Searches for text, and retrieves n words either side of the text, which are retuned seperatly'''
    word = r"\W*([\w]+)"
    print('TEXT:')
    print(text)

    age = get_age_search_word(text, "year", n)

    if age is None:
        age = get_age_search_word(text, "yo", n)

    if age is None:
        age = get_age_search_word(text, "y/o", n)

    if age is None:
        age = get_age_search_word(text, "age", n)

    if age is None:
        age = re.search(r'\d{2} F', text, re.IGNORECASE)
        if age is None:
            age = re.search(r'\d{2}F', text, re.IGNORECASE)

        if age is not None:
            age = age.group(0)[:2]

    if age is None:
        age = re.search(r'\d{2} M', text, re.IGNORECASE)
        if age is None:
            age = re.search(r'\d{2}M', text, re.IGNORECASE)

        if age is not None:
            age = age.group(0)[:2]

    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGEEEEEEEEEEEEEEEEEEEEEEEEEEE', age)
    return age


def get_age_search_word(text, search_word, n):
    print('text ', text)
    print('word', search_word)

    word = r"\W*([\w]+)"
    age = re.search(r'{}\W*{}{}'.format(word * n, search_word, word * n), text, re.IGNORECASE)
    if age is not None:
        age = re.search(r'\d+', age.group())
    if age is not None:
        return age.group()

    return None
