"""
Contains a function to form balanced training set using SMOTE
"""
from balance_train_data.vectorize_text import get_vectorized_train_test
from imblearn.over_sampling import SMOTE


def get_smote_data(train, test, ngram_min, ngram_max):
    """ :param train: pandas dataset of train data
        :param test: pandas dataset of test data
        :param ngram_min: ngram start index
        :param ngram_max: ngram end index
        :return: vectorized and balanced train and test datasets

    To balance text data using SMOTE, vectorization has to be done first.
    """
    # return term-document matrix. Treat text as matrices
    train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)

    sm = SMOTE()
    # resample the dataset
    train_TEXT, train_OUTPUT = sm.fit_sample(train_TEXT, train.OUTPUT)

    return train_TEXT, train_OUTPUT, test_TEXT
