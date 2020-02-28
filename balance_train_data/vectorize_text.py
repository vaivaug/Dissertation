from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk import word_tokenize

global vectorizer


def get_vectorized_train_test(train, test, ngram_min, ngram_max):
    """

    :param train:
    :param test:
    :param ngram_min:
    :param ngram_max:
    :return:
    """
    global vectorizer

    vectorizer = CountVectorizer(max_features=3000, tokenizer=get_tokenizer, stop_words=get_stop_words(),
                                 ngram_range=(ngram_min, ngram_max))

    print("this can take longer")

    # create term-document matrices
    train_TEXT = vectorizer.fit_transform(train.TEXT.values)
    test_TEXT = vectorizer.transform(test.TEXT.values)

    return train_TEXT, test_TEXT


def get_tokenizer(text):
    t = str.maketrans(dict.fromkeys(string.punctuation + '0123456789', " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens


def get_stop_words():
    my_stop_words = ['the', 'and', 'to', 'of', 'was', 'with', 'a', 'on', 'in', 'for', 'name', 'is',
                     'patient', 's', 'he', 'at', 'as', 'or', 'one', 'she', 'his', 'her', 'am',
                     'were', 'you', 'pt', 'pm', 'by', 'be', 'had', 'your', 'this', 'date', 'from',
                     'there', 'an', 'that', 'p', 'are', 'have', 'has', 'h', 'but', 'o', 'namepattern',
                     'which', 'every', 'also', 'should', 'if', 'it', 'been', 'b', 'w', 'who', 'during',
                     'any', 'c', 'd', 'x', 'i', 'all', 'please']
    return my_stop_words


def get_feature_names():

    global vectorizer
    return vectorizer.get_feature_names()
