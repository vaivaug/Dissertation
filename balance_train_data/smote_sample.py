"""
Contains a function to form balanced training set using SMOTE
"""
from balance_train_data.vectorize_text import get_vectorized_train_test
from imblearn.over_sampling import SMOTE


def get_smote_data(train, test, ngram_min, ngram_max):
    """ To balance text data using SMOTE, vectorization has to be done first.

    @param train: pandas dataframe storing data used for training
    @param test: pandas dataframe storing data used for testing
    @param ngram_min: integer, minimum number of adjacent words to be used for vectorization
    @param ngram_max: integer, maximum number of adjacent words to be used for vectorization
    @return: train_TEXT: vectorized TEXT column of training data
             test_TEXT: vectorized TEXT column of test data
             train_OUTPUT: output of the new training data
    """

    # return term-document matrix. Treat text as matrices
    train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)

    sm = SMOTE()
    # resample the dataset
    train_TEXT, train_OUTPUT = sm.fit_sample(train_TEXT, train.OUTPUT)

    return train_TEXT, train_OUTPUT, test_TEXT
