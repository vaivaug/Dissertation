from sklearn.feature_extraction.text import CountVectorizer
from process_text import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
import collections

global train_TEXT, test_TEXT, train_OUTPUT, test_OUTPUT, model, predicted_OUTPUT, list_words


def get_test_predicted_OUTPUT(train, test, threshold, smote):

    global train_TEXT, test_TEXT, train_OUTPUT, test_OUTPUT, model, predicted_OUTPUT, list_words

    # remove new line characters and nulls
    train = get_clean_text(train)
    test = get_clean_text(test)
    print('lengths:')
    print(len(train))
    print(len(test))
    '''
    import scipy.sparse as sp

    vectorizer = CountVectorizer(max_features=3000, tokenizer=get_tokenizer, stop_words=get_stop_words(), ngram_range=(1, 1))
    subject_vectors = vectorizer.fit_transform(train.TEXT.values)

   # body_vectorizer = CountVectorizer(max_features=3000, tokenizer=get_tokenizer, stop_words=get_stop_words(), ngram_range=(1, 1))
    #body_vectors = body_vectorizer.fit_transform(train.AGE)

    train_TEXT = sp.hstack([subject_vectors,
                            train['AGE']])

    test_TEXT = vectorizer.transform(test.TEXT.values)
    '''

    # vectorizer creation
    vectorizer = CountVectorizer(max_features=3000, tokenizer=get_tokenizer, stop_words=get_stop_words(), ngram_range=(1, 4))

    print("this can take longer")
    # learn the vocabulary dictionary
    vectorizer.fit_transform(train.TEXT.values)

    # return term-document matrix
    train_TEXT = vectorizer.fit_transform(train.TEXT.values)
    
    test_TEXT = vectorizer.transform(test.TEXT.values)

    train_OUTPUT = train.OUTPUT
    test_OUTPUT = test.OUTPUT

    if smote:
        sm = SMOTE()
        train_TEXT, train_OUTPUT = sm.fit_sample(train_TEXT, train_OUTPUT)

        print(train_OUTPUT.value_counts())

    # logistic regression
    model = LogisticRegression(C=0.0001, penalty='l2')
    model.fit(train_TEXT, train_OUTPUT)

    list_words = vectorizer.get_feature_names()

    predicted_OUTPUT = np.where(model.predict_proba(test_TEXT)[:, 1] > threshold, 1, 0)


    print('******************text*****************')
    print(test)
    test.TEXT.to_csv('../TEST_TEXT.csv')

    print('******************TRUE OUTPUT*****************')
    test.OUTPUT.to_csv('../TEST_REAL_OUTPUT.csv')

    print(test.OUTPUT.iloc[[2]])
    print('******************PREDICTED OUTOUT*********************')
    print('predicted output: ', predicted_OUTPUT)
    print(type(predicted_OUTPUT))
    np.savetxt("../test_predicted_output.csv", predicted_OUTPUT, delimiter=",")

    get_sick_predicted_not_sick(test, predicted_OUTPUT)

    return test_OUTPUT, predicted_OUTPUT



def get_sick_predicted_not_sick(test, predicted_OUTPUT):
    sick_predicted_not_sick = 0
    sick_predicted_sick = 0
    not_sick_predicted_not_sick = 0
    not_sick_predicted_sick = 0
    print('length of test: ', len(test.index))
    print('length of predicted output: ', len(predicted_OUTPUT))
    row_number = 0
    for index, row in test.iterrows():
        if row['OUTPUT'] == 1 and predicted_OUTPUT[row_number] == 0:
            sick_predicted_not_sick+=1
            print('missed: ')
            print(row)
            print(row['HADM_ID'])
            print(row['TEXT'])
        elif row['OUTPUT'] == 1 and predicted_OUTPUT[row_number] == 1:
            sick_predicted_sick +=1
        elif row['OUTPUT'] == 0 and predicted_OUTPUT[row_number] == 0:
            not_sick_predicted_not_sick += 1
        elif row['OUTPUT'] == 0 and predicted_OUTPUT[row_number] == 1:
            not_sick_predicted_sick += 1
        row_number += 1

    print('sick_predicted_not_sick: ', sick_predicted_not_sick)
    print("sick_predicted_sick: ", sick_predicted_sick)
    print('not_sick_predicted_not_sick: ', not_sick_predicted_not_sick)
    print('not_sick_predicted_sick: ', not_sick_predicted_sick)
    print('sum: ', not_sick_predicted_sick+not_sick_predicted_not_sick+sick_predicted_sick+sick_predicted_not_sick)

def plot_AUC(test_OUTPUT):
    # no skill prediction
    ns_probs = [0 for _ in range(len(test_OUTPUT))]

    # keep probabilities for the positive outcome only
    lr_probs = model.predict_proba(test_TEXT)[:, 1]

    # calculate scores
    ns_auc = roc_auc_score(test_OUTPUT, ns_probs)
    lr_auc = roc_auc_score(test_OUTPUT, lr_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, thresholds1 = roc_curve(test_OUTPUT, ns_probs)
    lr_fpr, lr_tpr, thresholds2 = roc_curve(test_OUTPUT, lr_probs)
    train_fpr, train_tpr, threshold3 = roc_curve(train_OUTPUT, model.predict_proba(train_TEXT)[:, 1])

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No skills')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


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

    # join words with weight values
    joined_word_weight = dict(zip(list_words, weights[0]))

    # sort words by weight from the lowest to the highest i.e. from negative to positive importance
    sorted_word_weight = {k: v for k, v in sorted(joined_word_weight.items(),
                                                  key=lambda item: item[1])}
    return sorted_word_weight


def plot_one_side_importance(importance_dict):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    ax.barh(range(len(importance_dict)), list(importance_dict.values()), align='center')
    ax.set_yticks(range(len(importance_dict)))
    ax.set_yticklabels(list(importance_dict.keys()))
    ax.invert_yaxis()
    ax.set_xlabel('words')
    ax.set_title('Importance of words')
    plt.show()
    ax.invert_yaxis()
