"""
This file is accessed from the Main.py or Gui.py files to run the program end to end (data cleaning, training,
predicting, visualising the results)
"""
import metrics as metrics
from balance_train_data.vectorize_text import get_vectorized_train_test
from data_preparation.prepare_data import get_clean_dataframe
from data_preparation.train_test_data import get_train_test_datasets
from balance_train_data.sub_sampling_negatives import get_sub_sampling_negatives_data
import nltk
nltk.download('punkt')
from models.LogisticRegression import get_predicted_on_test_LR, plot_word_importance, get_predicted_on_train_LR
from evaluation.auc import plot_AUC
from evaluation.confusion_matrix import *
from balance_train_data.over_sampling_positives import get_over_sampling_positives_data
from balance_train_data.smote_sample import get_smote_data


# variable to check if the model is run on validation or test set
results_on_validation = False


def predict_test_validation_set(threshold, balancing_type, solver, ngram_min, ngram_max):
    """ Get clean data, call run_model_on_balanced_data function and evaluate the predictions

    @param threshold: a float number between 0 and 1, decision threshold parameter
    @param balancing_type: a string, represents the selected data balancing type
    @param solver: a string, represents the selected solver used for the Linear Regression model
    @param ngram_min: integer, minimum number of adjacent words to be used for vectorization
    @param ngram_max: integer, maximum number of adjacent words to be used for vectorization
    """

    # read and clean input data
    notes_adm = get_clean_dataframe()

    print(notes_adm['OUTPUT'].value_counts())

    # all data split into train, test and validation sets
    # treat validation set as test set for now
    # do not touch test set till the end now
    train, test, validation = get_train_test_datasets(notes_adm)

    if results_on_validation:
        predicted_OUTPUT, prediction_probs = balance_and_run_test_valid_LR(balancing_type,
                                                                train,
                                                                validation,
                                                                threshold,
                                                                solver,
                                                                ngram_min,
                                                                ngram_max)

        return validation.OUTPUT, predicted_OUTPUT, prediction_probs
    else:
        predicted_OUTPUT, prediction_probs = balance_and_run_test_valid_LR(balancing_type,
                                                                train,
                                                                test,
                                                                threshold,
                                                                solver,
                                                                ngram_min,
                                                                ngram_max)

        return test.OUTPUT, predicted_OUTPUT, prediction_probs


def predict_cross_val_train_set(threshold, balancing_type, solver, ngram_min, ngram_max):
    """ Get clean data, call run_model_on_balanced_data function and evaluate the predictions

    @param threshold: a float number between 0 and 1, decision threshold parameter
    @param balancing_type: a string, represents the selected data balancing type
    @param solver: a string, represents the selected solver used for the Linear Regression model
    @param ngram_min: integer, minimum number of adjacent words to be used for vectorization
    @param ngram_max: integer, maximum number of adjacent words to be used for vectorization
    """

    # read and clean input data
    notes_adm = get_clean_dataframe()

    # all data split into train, test and validation sets
    train, test, validation = get_train_test_datasets(notes_adm)

    train_OUTPUT, predicted_OUTPUT, prediction_probs = balance_and_run_train_LR(balancing_type,
                                                                                train,
                                                                                test,
                                                                                threshold,
                                                                                solver,
                                                                                ngram_min,
                                                                                ngram_max)

    return train_OUTPUT, predicted_OUTPUT, prediction_probs


def balance_and_run_test_valid_LR(balancing_type, train, test, threshold, solver, ngram_min, ngram_max):
    """ Balance the dataset: SMOTE first vectorizes the dataset and then balances it while the other two types
    balance the dataset and then vectorize it.
    Call get_test_predicted_OUTPUT function to train the model and get predictions on the test set

    @param balancing_type: a string, represents the selected data balancing type
    @param train: pandas dataframe, stores the data used for training the model
    @param test: pandas dataframe, stores the data used for testing
    @param threshold: a float number between 0 and 1, decision threshold parameter
    @param solver: a string, represents the selected solver used for the Linear Regression model
    @param ngram_min: integer, minimum number of adjacent words to be used for vectorization
    @param ngram_max: integer, maximum number of adjacent words to be used for vectorization
    @return: test_OUTPUT: list of output values in the test set
             predicted_OUTPUT: list of predicted output values for the test set
             prediction_probs: list of prediction probabilities for the test set, needed to plot the AUC
    """

    # balance train data set
    if balancing_type == "SMOTE":

        # vectorize the TEXT column and then balance the train data
        train_TEXT, train_OUTPUT, test_TEXT = get_smote_data(train, test, ngram_min, ngram_max)

    else:

        # balance the train data and then vectorize the TEXT column
        if balancing_type == "sub-sample negatives":
            train = get_sub_sampling_negatives_data(train)

        elif balancing_type == "over-sample positives":
            train = get_over_sampling_positives_data(train)

        train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)
        train_OUTPUT = train.OUTPUT

    print(train_OUTPUT.value_counts())

    predicted_OUTPUT, prediction_probs = get_predicted_on_test_LR(train_TEXT,
                                                                      train_OUTPUT,
                                                                      test_TEXT,
                                                                      threshold=threshold,
                                                                      solver=solver)

    return predicted_OUTPUT, prediction_probs


def balance_and_run_train_LR(balancing_type, train, test, threshold, solver, ngram_min, ngram_max):
    """ Balance the dataset: SMOTE first vectorizes the dataset and then balances it while the other two types
    balance the dataset and then vectorize it.
    Call get_test_predicted_OUTPUT function to train the model and get predictions on the test set

    @param balancing_type: a string, represents the selected data balancing type
    @param train: pandas dataframe, stores the data used for training the model
    @param test: pandas dataframe, stores the data used for testing
    @param threshold: a float number between 0 and 1, decision threshold parameter
    @param solver: a string, represents the selected solver used for the Linear Regression model
    @param ngram_min: integer, minimum number of adjacent words to be used for vectorization
    @param ngram_max: integer, maximum number of adjacent words to be used for vectorization
    @return: test_OUTPUT: list of output values in the test set
             predicted_OUTPUT: list of predicted output values for the test set
             prediction_probs: list of prediction probabilities for the test set, needed to plot the AUC
    """

    # balance train data set
    if balancing_type == "SMOTE":

        # vectorize the TEXT column and then balance the train data
        train_TEXT, train_OUTPUT, test_TEXT = get_smote_data(train, test, ngram_min, ngram_max)

    else:

        # balance the train data and then vectorize the TEXT column
        if balancing_type == "sub-sample negatives":
            train = get_sub_sampling_negatives_data(train)

        elif balancing_type == "over-sample positives":
            train = get_over_sampling_positives_data(train)

        train_TEXT, test_TEXT = get_vectorized_train_test(train, test, ngram_min, ngram_max)
        train_OUTPUT = train.OUTPUT

    print(train_OUTPUT.value_counts())
    # create model
    predicted_OUTPUT, prediction_probs = get_predicted_on_train_LR(train_TEXT,
                                                                       train_OUTPUT,
                                                                       threshold=threshold,
                                                                       solver=solver)

    return train_OUTPUT, predicted_OUTPUT, prediction_probs


def plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type,
                            solver, threshold, ngram_min, ngram_max):
    """ Plot positive and negative word importance, plot confusion matrix, area under the ROC curve,
    print the accuracy, precision and recall scores

    @param test_OUTPUT: list of output values in the test set
    @param predicted_OUTPUT: list of predicted output values for the test set
    @param prediction_probs: list of prediction probabilities for the test set, needed to plot the AUC
    """

    # illustrate word importance
    plot_word_importance()

    # evaluating the model
    cnf_matrix = get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)
    plot_confusion_matrix(cnf_matrix)

    # illustrate area under the ROC curve
    plot_AUC(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, threshold, ngram_min, ngram_max)
