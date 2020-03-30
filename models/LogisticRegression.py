"""
Create a Logistic Regression model and make predictions on test data.
Plot word importance for this model
"""
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as word_plt
import numpy as np
from balance_train_data.vectorize_text import get_feature_names
from sklearn.model_selection import cross_val_predict
import statsmodels.api as sm
global model


def get_predicted_on_test_LR(train_TEXT, train_OUTPUT, test_TEXT, threshold, solver):
    """Create Logistic Regression model on the train data. Calculate probability of having lung cancer for each patient
    Classify patients to positives and negatives depending on the threshold

    @param train_TEXT: TEXT column of train dataframe
    @param train_OUTPUT: OUTPUT column of train dataframe
    @param test_TEXT: TEXT column of test dataframe
    @param threshold: threshold value
    @param solver: type of solver for Logistic Regression
    @return: predicted_OUTPUT: list of 0 and 1 predictions for each row in the test set
             prediction_probs: list of probabilities between 0 and 1 for each row in the test set
    """
    global model

    # logistic regression
    model = LogisticRegression(C=0.0001, penalty='l2', solver=solver)

    # fit the model with training data. return fitted estimator
    model.fit(train_TEXT, train_OUTPUT)

    # get positive prediction probabilities
    prediction_probs = model.predict_proba(test_TEXT)[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_OUTPUT = np.where(prediction_probs > threshold, 1, 0)

    return predicted_OUTPUT, prediction_probs


def get_predicted_on_train_LR(train_TEXT, train_OUTPUT, threshold, solver):
    """Create Logistic Regression model on the train data. Calculate probability of having lung cancer for each patient
    Classify patients to positives and negatives depending on the threshold

    @param train_TEXT: TEXT column of train dataframe
    @param train_OUTPUT: OUTPUT column of train dataframe
    @param test_TEXT: TEXT column of test dataframe
    @param threshold: threshold value
    @param solver: type of solver for Logistic Regression
    @return: predicted_OUTPUT: list of 0 and 1 predictions for each row in the test set
             prediction_probs: list of probabilities between 0 and 1 for each row in the test set
    """
    global model

    # logistic regression
    model = LogisticRegression(C=0.0001, penalty='l2', solver=solver)

    predicted_probs = cross_val_predict(model, train_TEXT, train_OUTPUT, cv=5, method='predict_proba')
    predicted_probs = predicted_probs[:, 1]

    # classify samples into two classes depending on the probabilities
    predicted_OUTPUT = np.where(predicted_probs > threshold, 1, 0)

    return predicted_OUTPUT, predicted_probs


def plot_word_importance():
    """ Plot the importance of words when making a positive prediction and negative prediction
    """

    sorted_word_weight = get_sorted_word_importance_dict()

    positive_importance = {}
    negative_importance = {}

    for word in list(sorted_word_weight)[0:30]:
        negative_importance[word] = sorted_word_weight[word]

    for word in list(reversed(list(sorted_word_weight)))[0:30]:
        positive_importance[word] = sorted_word_weight[word]

    plot_one_side_importance(positive_importance, "plots/positive.png")
    plot_one_side_importance(negative_importance, "plots/negative.png")


def get_sorted_word_importance_dict():
    """ Form a dictionary of words (or groups of adjacent words depending on the selected ngram) and importance values

    @return: a dictionary of word-value pairs. Values are associated with the importance
        of the words given the model. Dictionary is sorted by value in an increasing order
    """

    # weights associated to words in list_words
    weights = model.coef_

    list_words = get_feature_names()

    # join words with weight values
    joined_word_weight = dict(zip(list_words, weights[0]))

    # sort words by weight from the lowest to the highest i.e. from negative to positive importance
    sorted_word_weight = {k: v for k, v in sorted(joined_word_weight.items(),
                                                  key=lambda item: item[1])}
    return sorted_word_weight


def plot_one_side_importance(importance_dict, image_filedir):
    """ Draw the word importance diagram when the word-value dictionary is given

    @param importance_dict: 30 most important (pos or neg) words sorted by the importance value
    @param image_filedir: directory where the diagram is saved using png format
    """
    word_plt.rcdefaults()
    fig, ax = word_plt.subplots()

    ax.barh(range(len(importance_dict)), list(importance_dict.values()), align='center')
    ax.set_yticks(range(len(importance_dict)))
    ax.set_yticklabels(list(importance_dict.keys()))
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Importance of words')
    word_importance_fig = word_plt.gcf()
    word_importance_fig.tight_layout()

    word_plt.draw()
    word_importance_fig.savefig(image_filedir)
