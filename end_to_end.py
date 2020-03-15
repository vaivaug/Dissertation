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


# variable to check if the model is run on validation or test set
results_on_validation = True


def run_end_to_end(threshold, balancing_type, solver, ngram_min, ngram_max):

    # read and clean input data
    notes_adm = get_clean_dataframe()

    # all data split into train, test and validation sets
    # treat validation set as test set for now
    # do not touch test set till the end now
    train, test, validation = get_train_test_datasets(notes_adm)

    if results_on_validation:
        test_OUTPUT, predicted_OUTPUT, prediction_probs = run_model_on_balanced_data(balancing_type,
                                                                                     train,
                                                                                     validation,
                                                                                     threshold,
                                                                                     solver,
                                                                                     ngram_min,
                                                                                     ngram_max)
    else:
        test_OUTPUT, predicted_OUTPUT, prediction_probs = run_model_on_balanced_data(balancing_type,
                                                                                     train,
                                                                                     test,
                                                                                     threshold,
                                                                                     solver,
                                                                                     ngram_min,
                                                                                     ngram_max)

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


def run_model_on_balanced_data(balancing_type, train, test, threshold, solver, ngram_min, ngram_max):

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

    # create model
    test_OUTPUT, predicted_OUTPUT, prediction_probs = get_test_predicted_OUTPUT(train_TEXT,
                                                                                train_OUTPUT,
                                                                                test_TEXT,
                                                                                test.OUTPUT,
                                                                                threshold=threshold,
                                                                                solver=solver)
    return test_OUTPUT, predicted_OUTPUT, prediction_probs


def append_results_file(filedir, threshold, solver, ngram_min, ngram_max):
    with open(filedir, 'a') as fd:
        fd.write()