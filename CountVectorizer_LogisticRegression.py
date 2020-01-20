from sklearn.feature_extraction.text import CountVectorizer
from process_text import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np

global train_TEXT, test_TEXT, train_OUTPUT, test_OUTPUT, model, predicted_OUTPUT


def get_test_predicted_OUTPUT(train, test):

    global train_TEXT, test_TEXT, train_OUTPUT, test_OUTPUT, model, predicted_OUTPUT

    # remove new line characters and nulls
    train = get_clean_text(train)
    test = get_clean_text(test)

    # vectorizer creation
    vectorizer = CountVectorizer(max_features=3000, tokenizer=get_tokenizer, stop_words=get_stop_words())

    print("this can take longer")
    vectorizer.fit_transform(train.TEXT.values)

    train_TEXT = vectorizer.transform(train.TEXT.values)
    test_TEXT = vectorizer.transform(test.TEXT.values)
    print(type(test_TEXT))

    train_OUTPUT = train.OUTPUT
    test_OUTPUT = test.OUTPUT

    # logistic regression
    model = LogisticRegression(C=0.0001, penalty='l2')
    model.fit(train_TEXT, train_OUTPUT)

    # train_OUTPUT_preds = model.predict_proba(train_TEXT)[:, 1]
    # test_OUTPUT_preds = model.predict_proba(test_TEXT)[:, 1]

    predicted_OUTPUT = model.predict(test_TEXT)

    print('******************text*****************')
    test.TEXT.to_csv('../TEST_TEXT.csv')

    print('******************TRUE OUTPUT*****************')
    test.OUTPUT.to_csv('../TEST_REAL_OUTPUT.csv')

    print(test.OUTPUT.iloc[[2]])
    print('******************PREDICTED OUTOUT*********************')
    np.savetxt("../test_predicted_output.csv", predicted_OUTPUT, delimiter=",")
    return test_OUTPUT, predicted_OUTPUT


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

