"""
Create a Logistic Regression model and make predictions on test data.
Plot word importance for this model
"""
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as word_plt
import numpy as np
from balance_train_data.vectorize_text import get_feature_names

global model


def get_test_predicted_OUTPUT(train_TEXT, train_OUTPUT, test_TEXT, test_OUTPUT, threshold):
    """ :param train_TEXT: TEXT column of train dataframe
        :param train_OUTPUT: OUTPUT column of train dataframe
        :param test_TEXT: TEXT column of test dataframe
        :param test_OUTPUT: OUTPUT column of test dataframe
        :param threshold: threshold value

    Create Logistic Regression model on the train data. Calculate probability of having lung cancer for each patient
    Classify patients to positives and negatives depending on the threshold
    """
    global model

    # logistic regression
    model = LogisticRegression(C=0.0001, penalty='l2')

    # fit the model with training data. return fitted estimator
    model.fit(train_TEXT, train_OUTPUT)

    # get positive prediction probabilities
    prediction_probs = model.predict_proba(test_TEXT)[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_OUTPUT = np.where(prediction_probs > threshold, 1, 0)

    return test_OUTPUT, predicted_OUTPUT, model, prediction_probs


def plot_word_importance():
    """
    Plot the importance of words when making a positive prediction and negative prediction
    """

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
    """ :return: a dictionary of word-value pairs. Values are associated with an importance
        of the words given the model

    """
    # weights associated to words in list_words
    weights = model.coef_
    abs_weights = np.abs(weights)
    print('WEIGHTS:')
    print(abs_weights)

    list_words = get_feature_names()

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
   # word_plt.savefig('plots/word_importance_plt.png')
  #  word_plt.clf()

