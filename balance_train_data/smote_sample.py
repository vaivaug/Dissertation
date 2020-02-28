"""
Contains a function to form balanced training set using SMOTE
"""
from balance_train_data.vectorize_text import get_vectorized_train_test
from imblearn.over_sampling import SMOTE


def get_smote_data(train, test, ngram_min, ngram_max):
    """ :param train: pandas dataset of train data
        :param test: pandas dataset of test data
    """
    # return term-document matrix. Treat text as matrices
    train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)
    sm = SMOTE()
    train_TEXT, train_OUTPUT = sm.fit_sample(train_TEXT, train.OUTPUT)

    return train_TEXT, train_OUTPUT, test_TEXT
