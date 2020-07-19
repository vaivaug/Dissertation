"""
Create a Logistic Regression model and make predictions on test data.
Plot word importance for this model
"""
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as word_plt
import numpy as np
from balance_train_data.vectorize_text import get_feature_names
from sklearn.model_selection import cross_val_predict
global model
import pickle
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from balance_train_data.over_sampling_positives import get_over_sampling_positives_data
from balance_train_data.sub_sampling_negatives import get_sub_sampling_negatives_data
from balance_train_data.smote_sample import get_smote_data
from balance_train_data.vectorize_text import get_vectorized_train_test


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

    # Save the trained model
    joblib.dump(model, 'saved_model.pkl')

    return predicted_OUTPUT, prediction_probs


def get_predicted_on_train_LR(train, threshold, solver, balancing_type, ngram_min, ngram_max):
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

    #predicted_probs = cross_val_predict(model, train_TEXT, train_OUTPUT, cv=5, method='predict_proba')
    #predicted_probs = predicted_probs[:, 1]

    train_TEXT = train[['TEXT']]
    train_OUTPUT = train[['OUTPUT']]
    print(train_TEXT)
    # KFold Cross Validation approach
    kf = KFold(n_splits=5, shuffle=False)
    kf.split(train_TEXT)

    # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
    predicted_probs = []

    # Iterate over each train-test split
    for train_index, test_index in kf.split(train_TEXT):
        # Split train-test
        model = LogisticRegression(C=0.0001, penalty='l2', solver=solver)

        X_train, X_test = train_TEXT.iloc[train_index], \
                          train_TEXT.iloc[test_index]
        y_train, y_test = train_OUTPUT.iloc[train_index], \
                          train_OUTPUT.iloc[test_index]

        print('sick people count in test set: ')
        print(y_test['OUTPUT'].value_counts())

        # balance train data
        X_train = pd.concat([X_train, y_train]).reindex(index=X_train.index, columns=X_train.columns)
       # X_train['OUTPUT'] = y_train['OUTPUT']
        X_train.reset_index(drop=True)

        X_test['OUTPUT'] = y_test['OUTPUT']
        X_test.reset_index(drop=True)

        print('eina 1')
        #print(new_train_dataset)
        #print(new_test_dataset)
        train_TEXT_part, train_OUTPUT_part, test_TEXT_part = get_balanced_data(balancing_type, X_train,
                                                                               X_test,
                                                                ngram_min, ngram_max)
        print('sick value count on train data when kfold: ')
        print(train_OUTPUT_part.value_counts())
        # Train the model
        model = model.fit(train_TEXT_part, train_OUTPUT_part)
        # Append to accuracy_model the accuracy of the model
        ##predicted_probs.extend(model.predict_proba(test_TEXT_part)[:, 1])

        #print('predicted ', predicted_probs)
        #print('actual for tis try ', y_test)
    print('praejo visus')
    # Print the accuracy
    print(type(predicted_probs))
    print('length of predicted_probs ', len(predicted_probs))

    # classify samples into two classes depending on the probabilities
    predicted_OUTPUT = np.where(predicted_probs > threshold, 1, 0)

    return predicted_OUTPUT, predicted_probs


def get_balanced_data(balancing_type, train, test, ngram_min, ngram_max):

    if balancing_type == "SMOTE":

        # vectorize the TEXT column and then balance the train data
        train_TEXT, train_OUTPUT, test_TEXT = get_smote_data(train, test, ngram_min, ngram_max)

    else:

        # balance the train data and then vectorize the TEXT column
        if balancing_type == "sub-sample negatives":
            train = get_sub_sampling_negatives_data(train)

        elif balancing_type == "over-sample positives":
            train = get_over_sampling_positives_data(train)

        print('eina2')
        train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)
        train_OUTPUT = train.OUTPUT

    return train_TEXT, train_OUTPUT, test_TEXT



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
