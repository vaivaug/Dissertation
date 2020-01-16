
import pandas as pd
import csv
from prepare_data import get_clean_dataframe
from train_test_data import get_train_test_datasets
from sub_sampling_negatives import get_sub_sampling_negatives_data
from process_text import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from CountVectorizer_LogisticRegression import *
from sklearn import metrics
from confusion_matrix import *

from nltk import word_tokenize
import string


notes_adm = get_clean_dataframe()
print(len(notes_adm))

train, test = get_train_test_datasets(notes_adm)

train = get_sub_sampling_negatives_data(train)


# remove new line characters and nulls
train = get_clean_text(train)
test = get_clean_text(test)


test_OUTPUT, predicted_OUTPUT = get_test_predicted_OUTPUT(train, test)

# confusion matrix
cnf_matrix = get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)
plot_confusion_matrix(cnf_matrix)

print("Accuracy:", metrics.accuracy_score(test_OUTPUT, predicted_OUTPUT))
print("Precision:", metrics.precision_score(test_OUTPUT, predicted_OUTPUT))
print("Recall:", metrics.recall_score(test_OUTPUT, predicted_OUTPUT))

plot_AUC()



