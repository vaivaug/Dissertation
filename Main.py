import metrics as metrics
from balance_train_data.vectorize_text import get_vectorized_train_test
from data_preparation.prepare_data import get_clean_dataframe
from data_preparation.train_test_data import get_train_test_datasets
from balance_train_data.sub_sampling_negatives import get_sub_sampling_negatives_data
import nltk
nltk.download('punkt')
from models.LogisticRegression import get_test_predicted_OUTPUT, plot_word_importance
from evaluation.auc import plot_AUC
from evaluation.confusion_matrix import *
from balance_train_data.over_sampling_positives import get_over_sampling_positives_data
from balance_train_data.smote_sample import get_smote_data


"""
Parameters 
"""
def run_main(threshold, balancing_type, model, ngram_min, ngram_max):

    # read and clean input data
    notes_adm = get_clean_dataframe()


    # all data split into train, test and validation sets
    # treat validation set as test set for now
    # do not touch test set till the end now
    train, test, validation = get_train_test_datasets(notes_adm)

    # balance train data set
    if balancing_type == "SMOTE":
        # vectorize the TEXT column and then balance the train data
        train_TEXT, train_OUTPUT, validation_TEXT = get_smote_data(train, validation, ngram_min, ngram_max)
    else:
        # balance the train data and then vectorize the TEXT column
        if balancing_type == "sub-sample negatives":
            train = get_sub_sampling_negatives_data(train)
        elif balancing_type == "over-sample positives":
            train = get_over_sampling_positives_data(train)

        train_TEXT, validation_TEXT = get_vectorized_train_test(train, validation, ngram_min, ngram_max)
        train_OUTPUT = train.OUTPUT

    # check that we have the same number of positives and negatives
    print('value counts: ', train.OUTPUT.value_counts())

    # create model
    test_OUTPUT, predicted_OUTPUT, model, prediction_probs = get_test_predicted_OUTPUT(train_TEXT,
                                                                                       train_OUTPUT,
                                                                                       validation_TEXT,
                                                                                       validation.OUTPUT,
                                                                                       threshold=threshold)

    # illustrate word importance
    plot_word_importance()

    # evaluating the model
    cnf_matrix = get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)
    plot_confusion_matrix(cnf_matrix)

    print("Accuracy:", metrics.accuracy_score(test_OUTPUT, predicted_OUTPUT))
    print("Precision:", metrics.precision_score(test_OUTPUT, predicted_OUTPUT))
    print("Recall:", metrics.recall_score(test_OUTPUT, predicted_OUTPUT))

    # illustrate area under the ROC curve
    plot_AUC(test_OUTPUT, prediction_probs)

    # last step, check how the model performs on the test data

