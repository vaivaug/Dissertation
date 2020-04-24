"""
Class used to vectorize text (create term-document matrices)
"""
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk import word_tokenize

global vectorizer


def get_vectorized_train_test(train, test, ngram_min, ngram_max):
    """ Create term-document matrices for the TEXT columns in train and test datasets

    @param train: pandas dataframe, stores the data used for training the model
    @param test: pandas dataframe, stores the data used for testing the model
    @param ngram_min: integer, minimum number of adjacent words to be used for vectorization
    @param ngram_max: integer, maximum number of adjacent words to be used for vectorization
    @return: train_TEXT: vectorized TEXT column for training data
             test_TEXT: vectorized TEXT column for test data
    """
    global vectorizer

    vectorizer = CountVectorizer(max_features=3000, tokenizer=get_tokenizer, stop_words=get_stop_words(),
                                 ngram_range=(ngram_min, ngram_max))

    # create term-document matrices
    train_TEXT = vectorizer.fit_transform(train.TEXT.values)
    test_TEXT = vectorizer.transform(test.TEXT.values)

    return train_TEXT, test_TEXT


def get_tokenizer(text):
    """ Text string split into word strings, clean words (no unnecessary characters, punctuation etc.)

    @param text: strings in the TEXT columns
    @return: tokens: words, tokenized text
    """

    t = str.maketrans(dict.fromkeys(string.punctuation + '0123456789', " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens


def get_stop_words():
    """ Store list of useless words, used for Vectorizing text

    @return: stop_words: list of stop words
    """
    stop_words = ['the', 'and', 'to', 'of', 'was', 'with', 'a', 'on', 'in', 'for', 'name', 'is',
                     'patient', 's', 'he', 'at', 'as', 'or', 'one', 'she', 'his', 'her', 'am',
                     'were', 'you', 'pt', 'pm', 'by', 'be', 'had', 'your', 'this', 'date', 'from',
                     'there', 'an', 'that', 'p', 'are', 'have', 'has', 'h', 'but', 'o', 'namepattern',
                     'which', 'every', 'also', 'should', 'if', 'it', 'been', 'b', 'w', 'who', 'during',
                     'any', 'c', 'd', 'x', 'i', 'all', 'please']
    return stop_words


def get_feature_names():
    """ Use the same vectorizer object to get the feature names. Used when plotting word importance

    @return: a list of feature names
    """
    global vectorizer
    return vectorizer.get_feature_names()
