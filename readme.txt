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


