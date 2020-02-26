
from prepare_data import get_clean_dataframe
from train_test_data import get_train_test_datasets
from sub_sampling_negatives import get_sub_sampling_negatives_data
import nltk
nltk.download('punkt')
from CountVectorizer_LogisticRegression import get_test_predicted_OUTPUT, plot_word_importance
from auc import plot_AUC
from confusion_matrix import *
from over_sampling_positives import get_over_sampling_positives_data


'''
1. joined discharge summaries    DONE
2. check which words have the biggest influence for 1 or 0 output prediction DONE
3. why empty discharge summary predicts lung cancer. Possibly some of the lung cancer diagnoses have am empty discharge sumary DONE 
4. add ngram_range=(1,3) inside CountVectorizer
5. look into which files are at the top right
6. remove empty ones from all data
for friday: age, sex, AUC, 
For AUC diagram, add diagram AUC = , Confidence intervals

7. create my own diagnosis dictionary
8. check if corrections are fair
'''

# read and clean input data
notes_adm = get_clean_dataframe()


# all data split into train, test and validation sets
# treat validation set as test set for now
# do not touch test set till the end now
train, test, validation = get_train_test_datasets(notes_adm)

# values of parameters to be pressed in UI
threshold = 0.5
smote_selected = False
sub_sample_negatives_selected = True
over_sample_positives_selected = False
ngram_min = 1
ngram_max = 1

# balance train dataset
if sub_sample_negatives_selected:
    train = get_sub_sampling_negatives_data(train)
elif over_sample_positives_selected:
    train = get_over_sampling_positives_data(train)

print('value counts: ', train.OUTPUT.value_counts())

test_OUTPUT, predicted_OUTPUT, model, prediction_probs = get_test_predicted_OUTPUT(train,
                                                                                   validation,
                                                                                   threshold=threshold,
                                                                                   smote=smote_selected,
                                                                                   ngram_min=ngram_min,
                                                                                   ngram_max=ngram_max)

plot_word_importance()

# confusion matrix
cnf_matrix = get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)

plot_confusion_matrix(cnf_matrix)
print("Accuracy:", metrics.accuracy_score(test_OUTPUT, predicted_OUTPUT))
print("Precision:", metrics.precision_score(test_OUTPUT, predicted_OUTPUT))
print("Recall:", metrics.recall_score(test_OUTPUT, predicted_OUTPUT))

plot_AUC(test_OUTPUT, prediction_probs)


'''
# print TEXT of specific patient given HADM_ID
for index, row in notes_adm.iterrows():
    if row['HADM_ID'] == 140248:
        print(row['HADM_ID'])
        print(row['TEXT'])
        break
'''

# print('number of cancer sick people: \n', notes_adm.OUTPUT.value_counts())
# print(notes_adm.loc[notes_adm['HADM_ID']==169684].to_string())
