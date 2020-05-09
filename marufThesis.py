import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import roc_auc_score
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#************Data Preprocessing part**********************
data = pd.read_csv("paper_review.csv")

#Checking dataframe upper portion and columns
# print(data.head())
# for col in data:
#      print(col)
# print(data[['review__evaluation']])

#checking us there any Nan value or not
# print(data['review__evaluation'].isnull().values.any())
# print(data['review__evaluation'].isnull().sum())
data['review__evaluation'] = data['review__evaluation'].fillna(method='bfill')
# print(data['review__text'].isnull().sum())
data['review__text'] = data['review__text'].fillna(method='bfill')

# print(data['review__evaluation'])


#combining two principle column which are necessary
data1 = data[['review__text','review__evaluation']]
# print(data1.head())
# print(data1['review__text'].size)
# print(data1["review__evaluation"].size)

#Plotting the Class labels and number of frequently occured
# fig = plt.figure(figsize=(8,6))
# # data1.groupby('review__evaluation').review__text.count().plot.bar(ylim=0)
# # plt.show()
# plt.figure(figsize=(8,9))
# ax=sns.countplot(x='review__evaluation',data=data1)
#
# labels = ['very negative','negative','neutral','positive','very positive']
# ax.set_xticklabels(labels)
#
# for p in ax.patches:
#     height = p.get_height()
#     if(np.isnan(height)):
#         height=0
#     ax.text(p.get_x()+p.get_width()/2.,
#             height + 3,
#             height,
#             ha="center")
#
# plt.show()


#Model creation

#importing stopwords for spanish language
final_stopwords_list = stopwords.words('spanish') + stopwords.words('english')
# one method to apply TF-IDF
# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=final_stopwords_list)

# Another methof to apply TF-IDF
tfidf = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=final_stopwords_list )
processed_features = tfidf.fit_transform(data1['review__text']).toarray()
# print(processed_features.shape)


#Splitting test dataset and train dataset
X_train, X_test, y_train, y_test = train_test_split(processed_features, data1['review__evaluation'], test_size=0.2, random_state=0)


#**********Naive Bayes classifier**************
from sklearn.naive_bayes import MultinomialNB
text_classifier = MultinomialNB().fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print("naive bayes classifier accuracy: ",accuracy_score(y_test, predictions))

#**************LinearSVC classifier***************
from sklearn.svm import LinearSVC
text_classifier = LinearSVC().fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print("LinearSVC classifier accuracy: ",accuracy_score(y_test, predictions))

#************Logistic regression Model*****************
from sklearn.linear_model import LogisticRegression
text_classifier = LogisticRegression(random_state=0)
text_classifier = text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print("Logistic regression Model accuracy: ",accuracy_score(y_test, predictions))

#******************KNN*****************************
from sklearn.neighbors import KNeighborsClassifier
text_classifier = KNeighborsClassifier(n_neighbors=5)
text_classifier = text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print("KNN Model accuracy: ",accuracy_score(y_test, predictions))

#*********************RandomForestClassifier****************
from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier1 = text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print("RandomForestClassifier accuracy: ",accuracy_score(y_test, predictions))

#*************Cross-validation*********************
from sklearn.model_selection import cross_val_score
models = [
    KNeighborsClassifier(n_neighbors=5),
    RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 10
final_result = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, processed_features, data1['review__evaluation'], scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
final_result  = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
final_result = final_result .groupby('model_name').accuracy.max()
print("Models with max accuracy after 10 fold cross validation")
print(final_result)
#
# '''
#     As the best model LinearSVC therefore further evaluation
# '''
#
text_classifier = LinearSVC().fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
print("LinearSVC classifier accuracy: ",accuracy_score(y_test, predictions))
print("LinearSVC confusion matrix")
print(confusion_matrix(y_test,predictions))
print("LinearSVC classification report")
print(classification_report(y_test,predictions))

#*******confusion matrix diagram*******
conf_mat = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#*************************ROC Curve***************************
n_classes = [-2,-1,0,1,2]
pred1=text_classifier.predict(X_test)
t1=sum(x==0 for x in pred1-y_test)/len(pred1)

### MACRO
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(n_classes)):
    fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(pred1))[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(n_classes))]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(len(n_classes)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= len(n_classes)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
plt.figure(figsize=(8,5))
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='green', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','black'])
for i, color in zip(range(len(n_classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.annotate('Random Guess',(.5,.48),color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for LinearSVC')
plt.legend(loc="lower right")
plt.show()



