"""
Contains a function which adds an AGE column to all the patients in the dataframe.
If the age is not extracted, column value is NaN
Age filter was not used at the end
"""
import numpy as np
import re
import datetime
from dateutil.relativedelta import relativedelta


def get_data_with_age_column(data):
    """ Create Age column for the given dataset. Age is mentioned in free text, no consistent structure.
    The following code extracts the age of all the patients apart from around 200. The code is not used in the final
    program run.

    @param data: pandas dataframe, containing rows of training or testing data sets
    @return: pandas dataframe, containing rows of training or testing data sets, added AGE column
    """

    # initialise age column with NaN
    data['AGE'] = np.nan

    # read
    # iterate through all the patients
    for index, row in data.iterrows():
        # get age from date of birth mentioned in TEXT
        age = get_age_from_date_of_birth(row['TEXT'])

        if age is None:
            # get age from the first few sentences in TEXT
            age = get_age_from_text(" ".join(row['TEXT'].split()[:200]), 3)

        data.set_value(index, 'AGE', age)

    return data


def get_age_from_date_of_birth(text):
    """ Get age by subtracting Date of Birth from the Discharge Date.
    If the 'Date of Birth' is mentioned, then extract the date from the next following words, find the end date,
    calculate the age.

    @param text: discharge summary string
    @return: integer number for age or None
    """

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
    """ Search for words that are written next to the person's age. The most common cases are searched first

    @param text: first 200 words of the patient's discharge summary (TEXT column)
    @param n: integer, number of words to search around the specified word
    @return: integer number for age or None
    """

    age = get_age_search_word(text, "year", n)

    if age is None:
        age = get_age_search_word(text, "yo", n)

    if age is None:
        age = get_age_search_word(text, "y/o", n)

    if age is None:
        age = get_age_search_word(text, "y.o", n)

    if age is None:
        age = get_age_search_word(text, "age", n)

    if age is None:
        # age mentioned next to F letter standing for Female
        age = re.search(r'\d{2} F', text, re.IGNORECASE)
        if age is None:
            age = re.search(r'\d{2}F', text, re.IGNORECASE)

        if age is not None:
            age = age.group(0)[:2]

    if age is None:
        # age mentioned next to M letter standing for Male
        age = re.search(r'\d{2} M', text, re.IGNORECASE)
        if age is None:
            age = re.search(r'\d{2}M', text, re.IGNORECASE)

        if age is not None:
            age = age.group(0)[:2]

    return age


def get_age_search_word(text, search_word, n):
    """
    @param text: first 200 words of the patient's discharge summary (TEXT column)
    @param search_word: string that is possibly written next to the person's age
    @param n: integer, number of words to search around the specified word
    @return: integer number for age or None
    """

    word = r"\W*([\w]+)"
    age = re.search(r'{}\W*{}{}'.format(word * n, search_word, word * n), text, re.IGNORECASE)
    if age is not None:
        age = re.search(r'\d+', age.group())
    if age is not None:
        return age.group()

    return None
