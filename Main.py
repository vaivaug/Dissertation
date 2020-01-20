
from prepare_data import get_clean_dataframe
from train_test_data import get_train_test_datasets
from sub_sampling_negatives import get_sub_sampling_negatives_data
import nltk
nltk.download('punkt')
from CountVectorizer_LogisticRegression import get_test_predicted_OUTPUT, plot_AUC, plot_word_importance
from confusion_matrix import *
from sick_ones_to_file import write_sick_ones_to_file

'''
1. joined discharge summaries    DONE
2. check which words have the biggest influence for 1 or 0 output prediction
3. why empty discharge summary predicts lung cancer. Possibly some of the lung cancer diagnoses have am empty discharge sumary
4. add ngram_range=(1,3) inside CountVectorizer 
'''
notes_adm = get_clean_dataframe()

# write_sick_ones_to_file('../sick_ones.csv', notes_adm)

print(len(notes_adm))

train, test = get_train_test_datasets(notes_adm)

train = get_sub_sampling_negatives_data(train)

test_OUTPUT, predicted_OUTPUT = get_test_predicted_OUTPUT(train, test)

plot_word_importance()
# confusion matrix
cnf_matrix = get_confusion_matrix(test_OUTPUT, predicted_OUTPUT)
plot_confusion_matrix(cnf_matrix)

print("Accuracy:", metrics.accuracy_score(test_OUTPUT, predicted_OUTPUT))
print("Precision:", metrics.precision_score(test_OUTPUT, predicted_OUTPUT))
print("Recall:", metrics.recall_score(test_OUTPUT, predicted_OUTPUT))

plot_AUC(test_OUTPUT)




