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
# False - run on test set; True - run on validation set
results_on_validation = False
filter_by_age = True
filter_multiple_diseases = True
age_min = 45
age_max = 1000

ages = [[45, 1000], [45, 80], [50, 1000], [50, 80]]


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

    # notes_adm.to_csv('some_output_to_test4.csv')

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

    global age_min, age_max
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

    if filter_by_age:
        for a in ages:
            age_min = a[0]
            age_max = a[1]
            confusion_matrix_age_filter(age_min, age_max, test, predicted_OUTPUT.copy())

    if filter_multiple_diseases:
        confusion_matrix_multiple_diseases(test, predicted_OUTPUT.copy(), 3)
        confusion_matrix_multiple_diseases(test, predicted_OUTPUT.copy(), 4)

    return predicted_OUTPUT, prediction_probs


def confusion_matrix_age_filter(age_min, age_max, test, predicted_OUTPUT_with_age):
    count = 0
    for i, row in test.iterrows():
        if (row['AGE'] < age_min or row['AGE'] > age_max) and predicted_OUTPUT_with_age[count] == 1:
            predicted_OUTPUT_with_age[count] = 0
        count += 1

    cnf_matrix = metrics.confusion_matrix(test.OUTPUT, predicted_OUTPUT_with_age)

    class_names = [0, 1]
    fig, ax = cnf_plt.subplots()
    tick_marks = np.arange(len(class_names))
    cnf_plt.xticks(tick_marks, class_names, fontsize=14)
    cnf_plt.yticks(tick_marks, class_names, fontsize=14)

    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 22}, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    cnf_plt.tight_layout()
    cnf_plt.title('Confusion Matrix', y=1.1, fontsize=14)
    cnf_plt.ylabel('Actual label', fontsize=14)
    cnf_plt.xlabel('Predicted label', fontsize=14)
    conf_matrix_fig = cnf_plt.gcf()
    conf_matrix_fig.tight_layout()
    cnf_plt.draw()
    conf_matrix_fig.savefig('plots/conf_matrix_age_{}_to_{}.png'.format(age_min, age_max))
    cnf_plt.clf()


def confusion_matrix_multiple_diseases(test, predicted_OUTPUT_multiple_diseases, number_of_diseases):
    count = 0
    for i, row in test.iterrows():
        if (len(row['DIAGNOSIS'].split(';'))) >= number_of_diseases and predicted_OUTPUT_multiple_diseases[count] == 1:
            predicted_OUTPUT_multiple_diseases[count] = 0
        count += 1

    cnf_matrix = metrics.confusion_matrix(test.OUTPUT, predicted_OUTPUT_multiple_diseases)

    class_names = [0, 1]
    fig, ax = cnf_plt.subplots()
    tick_marks = np.arange(len(class_names))
    cnf_plt.xticks(tick_marks, class_names, fontsize=14)
    cnf_plt.yticks(tick_marks, class_names, fontsize=14)

    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, annot_kws={"size": 22}, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    cnf_plt.tight_layout()
    cnf_plt.title('Confusion Matrix', y=1.1, fontsize=14)
    cnf_plt.ylabel('Actual label', fontsize=14)
    cnf_plt.xlabel('Predicted label', fontsize=14)
    conf_matrix_fig = cnf_plt.gcf()
    conf_matrix_fig.tight_layout()
    cnf_plt.draw()
    conf_matrix_fig.savefig('plots/conf_matrix_{}_diseases_dismissed.png'.format(number_of_diseases))
    cnf_plt.clf()


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
    '''
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
    '''

    predicted_OUTPUT, prediction_probs = get_predicted_on_train_LR(train, threshold, solver, balancing_type,
                                                                   ngram_min, ngram_max)

    return train.OUTPUT, predicted_OUTPUT, prediction_probs


def plot_evaluation_metrics(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type,
                            solver, threshold, ngram_min, ngram_max):
    """ Plot positive and negative word importance, plot confusion matrix, area under the ROC curve,
    print the accuracy, precision and recall scores

    @param test_OUTPUT: list of output values in the test set
    @param predicted_OUTPUT: list of predicted output values for the test set
    @param prediction_probs: list of prediction probabilities for the test set, needed to plot the AUC
    """

    # evaluating the model
    cnf_matrix = get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)
    plot_confusion_matrix(cnf_matrix, threshold, balancing_type, solver, ngram_min, ngram_max)

    # illustrate area under the ROC curve
    plot_AUC(test_OUTPUT, predicted_OUTPUT, prediction_probs, balancing_type, solver, threshold, ngram_min, ngram_max)

    # illustrate word importance
    plot_word_importance()
