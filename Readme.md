# Introduction
This is a practice file for sentiment analysis using machine learning models.
#Working procedure
This work contains sentiment analysis on spanish dataset.The data set consists of paper reviews sent to an international conference mostly in Spanish (some are in English).
In order to do that, I used various machine learning classifier models to compare result. For getting the best output sone sorts of data cleaning part were also done as there were some missing values as well as some punctuations.
#ML classifier list
*KNeighborsClassifier      
*LinearSVC                 
*LogisticRegression        
*MultinomialNB             
*RandomForestClassifier  
#Evaluation  
```
naive bayes classifier accuracy:  0.45121951219512196
LinearSVC classifier accuracy:  0.4024390243902439
Logistic regression Model accuracy:  0.47560975609756095
KNN Model accuracy:  0.2804878048780488
RandomForestClassifier accuracy:  0.36585365853658536
```
After running cross validation(max value taken):
```
model_name                Accuracy
KNeighborsClassifier      0.341463
LinearSVC                 0.463415
LogisticRegression        0.439024
MultinomialNB             0.414634
RandomForestClassifier    0.400000
```
LinearSVC confusion matrix
```
[[ 6  5  6  4  3]
 [ 2  0  1  1  3]
 [ 2  0  2  2  4]
 [ 1  0  3  8  4]
 [ 5  0  3  0 17]]
```
LinearSVC classification report
```
              precision    recall  f1-score   support

        -2.0       0.38      0.25      0.30        24
        -1.0       0.00      0.00      0.00         7
         0.0       0.13      0.20      0.16        10
         1.0       0.53      0.50      0.52        16
         2.0       0.55      0.68      0.61        25

    accuracy                           0.40        82
   macro avg       0.32      0.33      0.32        82
weighted avg       0.40      0.40      0.39        82

```
Though the result is not up to the mark, It is just a practice test further I will improve the accuarcy.
The ROC curve and confusion matrix diagram also prepared for visualization.
#Dataset
[Datasert link](https://archive.ics.uci.edu/ml/datasets/Paper+Reviews)
