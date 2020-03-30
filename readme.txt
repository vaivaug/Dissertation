Dissertation Progress Report

Working on the project has been consistent so far. I have been working with a US medical dataset from MIMIC-III Clinical Database. 
I had to prepare the data to be used for predicting patients at high risk of lung cancer. The process involved looking at the 'Diagnoses' column, removing the 'NEWBORN', selecting text from 'discharge summary' categories. 
After the data cleaning, followed text processing, tokenisation. 
It is worth pointing out that the data is very imbalanced. Therefore, I have tried three different techniques to deal with imbalanced datasets: over-sampling the positives, sub-sample the negatives and SMOTE. 
I am currently using CountVectorizer class and Logistic Regression to make predictions. The program produces some high accuracy results. However, I will try applying SVM, Random Forrest and possibly some other algorithms to see if the results are improved. 
I have had a few meetings with doctors in medical building. I received feedback and changed my code to reflect what has been discussed. 
In addition to the initial plan, I will create a simple UI to represent the model results. 

To sum up, the project is going well and my supervisor seems happy with the progress we made. 

Plan:
cross validation
check the best C value
check the best max_features value
decide on structure of experiments
run experiments for each type of imbalanced data
update trello board
write a report

Questions:
1. Run experiments 3*5 = 15 output files (3 different balancing types, 5 different solvers)
For each out of 15, we run different thresholds and ngrams.

2. thresholds: 0, 0.05, 0.1, 0.15, 0.2 ....,  0.95,  1 ? (every 0.05)
   ngrams:  (1, 1); (1, 2); (1, 3); (2, 2); (2, 3); (3, 3) ?    Do I need 4?

   run 5 random ones, if oversampling is a win for most of them, then

3. which evaluation parameters to store: AUC
plot states: parameters
store only AUC
confidence interval
percentage of false negative
output AUC plots, with some text

4. The best will be the one with the lowest FN, and quiet low top right corner? Keep percentages of both?

5. I need to have diagrams about the best C and max_features values, Do I do it before experiments or after the experiments,
with the best threshold and solver type
NOT
4. Using test, validation and train. Not touching test at all. treat validation as test.
So do the cross validation: on train ; ?

5. Do I need diagrams from the article, Learning Curve, run on training predictions and validation prediction

6. at which point do I run on test set

7. I can write the paper after.


Plan:
1. do the cross validation on train set, have separate function for running on validation or test sets
2. confidence interval to output picture




First, the article I am using
https://towardsdatascience.com/introduction-to-clinical-natural-language-processing-predicting-hospital-readmission-with-1736d52bc709
referst to 'test set' to be the one used at the end to check overfitting and 'validation set' to be the one used
earlier when evaluating model performance. Therefore, I use the same definitions for both.

Questions:
I have cross validation for training data implemented, however since the data balancing techniques are used before
the cross validation, model is overfitting. For example, when using over sampling positives and then running the cross
validation, identical rows can be in both, train and validation datasets.
Therefore, since I have a lot of data, it should be okay to fit the model with the train data,
run it against validation set

email tom:
explain that validation and test set differences.
explain why Im running experiments on validation set
show the picture fir experiments ask if its ok


For next meeting Friday April 3rd:
1. Should I do some data visualisation, e.g. number of words in the discharge summaries, length of discharge summaries.

