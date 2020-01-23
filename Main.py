
from prepare_data import get_clean_dataframe
from train_test_data import get_train_test_datasets
from sub_sampling_negatives import get_sub_sampling_negatives_data
import nltk
nltk.download('punkt')
from CountVectorizer_LogisticRegression import get_test_predicted_OUTPUT, plot_AUC, plot_word_importance
from confusion_matrix import *
from sick_ones_to_file import write_sick_ones_to_file
from over_sampling_positives import get_over_sampling_positives_data
from age import get_data_with_age_column

'''
1. joined discharge summaries    DONE
2. check which words have the biggest influence for 1 or 0 output prediction DONE
3. why empty discharge summary predicts lung cancer. Possibly some of the lung cancer diagnoses have am empty discharge sumary DONE 
4. add ngram_range=(1,3) inside CountVectorizer
5. look into which files are at the top right
6. remove empty ones from all data
for friday: age, sex, AUC, 
For AUC diagram, add diagram AUC = , Confidence intervals
'''

notes_adm = get_clean_dataframe()
notes_adm = get_data_with_age_column(notes_adm)
print(notes_adm.AGE.value_counts().to_csv('counts.csv'))
print('no age: ', notes_adm.isna().sum())

write_sick_ones_to_file('../sick_ones.csv', notes_adm)

print(len(notes_adm))

'''values of parameters to be pressed in UI'''
threshold = 0.4
smote_selected = False
sub_sample_negatives_selected = True
over_sample_positives_selected = False

# all data split into train and test
train, test = get_train_test_datasets(notes_adm)

if sub_sample_negatives_selected:
    train = get_sub_sampling_negatives_data(train)
elif over_sample_positives_selected:
    train = get_over_sampling_positives_data(train)

test_OUTPUT, predicted_OUTPUT = get_test_predicted_OUTPUT(train, test, threshold=threshold, smote=smote_selected)

plot_word_importance()
# confusion matrix
cnf_matrix = get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)
plot_confusion_matrix(cnf_matrix)

print("Accuracy:", metrics.accuracy_score(test_OUTPUT, predicted_OUTPUT))
print("Precision:", metrics.precision_score(test_OUTPUT, predicted_OUTPUT))
print("Recall:", metrics.recall_score(test_OUTPUT, predicted_OUTPUT))

plot_AUC(test_OUTPUT)




