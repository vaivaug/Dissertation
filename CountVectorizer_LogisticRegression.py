from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as word_plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy import sparse
from sklearn import metrics
from nltk import word_tokenize
import string
import pandas as pd
import collections

global model, vectorizer


def get_test_predicted_OUTPUT(train, test, threshold, smote, ngram_min, ngram_max):

    global model

    # return term-document matrix. Treat text as matrices
    train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)

    # logistic regression
    model = LogisticRegression(C=0.0001, penalty='l2')

    # fit the model with training data. return fitted estimator
    model.fit(train_TEXT, train.OUTPUT)

    # get positive prediction probabilities
    prediction_probs = model.predict_proba(test_TEXT)[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_OUTPUT = np.where(prediction_probs > threshold, 1, 0)

    return test.OUTPUT, predicted_OUTPUT, model, prediction_probs


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
    # learn the vocabulary dictionary
    vectorizer.fit_transform(train.TEXT.values)

    # create term-document matrices
    train_TEXT = vectorizer.fit_transform(train.TEXT.values)
    test_TEXT = vectorizer.transform(test.TEXT.values)

    return train_TEXT, test_TEXT


def plot_word_importance():

    sorted_word_weight = get_sorted_word_importance_dict()

    positive_importance = {}
    negative_importance = {}

    for word in list(sorted_word_weight)[0:30]:
        negative_importance[word] = sorted_word_weight[word]

    print(negative_importance)

    for word in list(reversed(list(sorted_word_weight)))[0:30]:
        positive_importance[word] = sorted_word_weight[word]

    print(positive_importance)

    plot_one_side_importance(positive_importance)
    plot_one_side_importance(negative_importance)


def get_sorted_word_importance_dict():

    # weights associated to words in list_words
    weights = model.coef_
    abs_weights = np.abs(weights)
    print('WEIGHTS:')
    print(abs_weights)

    list_words = vectorizer.get_feature_names()

    # join words with weight values
    joined_word_weight = dict(zip(list_words, weights[0]))

    # sort words by weight from the lowest to the highest i.e. from negative to positive importance
    sorted_word_weight = {k: v for k, v in sorted(joined_word_weight.items(),
                                                  key=lambda item: item[1])}
    return sorted_word_weight


def plot_one_side_importance(importance_dict):
    word_plt.rcdefaults()
    fig, ax = word_plt.subplots()

    ax.barh(range(len(importance_dict)), list(importance_dict.values()), align='center')
    ax.set_yticks(range(len(importance_dict)))
    ax.set_yticklabels(list(importance_dict.keys()))
    ax.invert_yaxis()
    ax.set_xlabel('words')
    ax.set_title('Importance of words')
    word_plt.show()
    # ax.invert_yaxis()
    word_plt.savefig('plots/word_importance_plt.png')
    word_plt.clf()


def get_tokenizer(text):
    t = str.maketrans(dict.fromkeys(string.punctuation + '0123456789', " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens


def get_stop_words():
    my_stop_words = ['the', 'and', 'to', 'of', 'was', 'with', 'a', 'on', 'in', 'for', 'name',
                    'is', 'patient', 's', 'he', 'at', 'as', 'or', 'one', 'she', 'his', 'her', 'am',
                         'were', 'you', 'pt', 'pm', 'by', 'be', 'had', 'your', 'this', 'date',
                         'from', 'there', 'an', 'that', 'p', 'are', 'have', 'has', 'h', 'but', 'o',
                         'namepattern', 'which', 'every', 'also', 'should', 'if', 'it', 'been',
                         'who', 'during', 'any', 'c', 'd', 'x', 'i', 'all', 'please']
    return my_stop_words

'''
    if smote:
        print('goes')
        sm = SMOTE()
        train_TEXT, train_OUTPUT = sm.fit_sample(train_TEXT, train.OUTPUT)
       # train_TEXT = pd.DataFrame(train_TEXT.todense())
        print(type(train_TEXT))
       # train_TEXT.to_csv('smote_examples.csv')
        print(train_TEXT)
        print('smote value counts: ', train_OUTPUT.value_counts())

        # return term-document matrix. Treat text as matrices
        train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)
'''