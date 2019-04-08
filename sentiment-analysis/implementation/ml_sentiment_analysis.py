"""
Created on Tue March 23 4:45pm 2019
@author: De Jong Yeong (T00185309)

Machine Learning approach for sentiment analysis. It extracts feature with TF-IDF model to perform sentiment analysis.
TF-IDF: Term Frequency - Inverse Document Frequency. It is a product of two metrics (tfidf = tf * idf).

The review.tokened column contains normalized data.

Reference: Text Analysis with Python by Dipanjan Sarkar

Output file: -
"""

# import library
import numpy as np
import pandas as pd
import sklearn.metrics as mt
from tabulate import tabulate
from matplotlib import pyplot as plt
import sklearn.model_selection as ms
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# load data
filename = '../datasets/amazon_unlocked_mobile_datasets_with_sentiment.csv'
names = ['product.name', 'brand.name', 'review.text', 'review.process', 'review.tokened', 'score', 'sentiment']
fields = ['review.tokened', 'sentiment']
review = pd.read_csv(filename, names=names, usecols=fields)

print()

# split data 70% train 30% test
print('split data --- start')
array = review.values
X = array[:, 0:1]
Y = array[:, 1]
size = 0.3

# testX and trainX is review.tokened
# testY and trainY is sentiment
trainX, testX, trainY, testY = ms.train_test_split(X, Y, test_size=size, shuffle=True)
print('split data --- end')

print()

# feature extraction
# build tfidf feature on train reviews
# min_df ignore terms that have a document frequency lower than threshold when building vocab
# max_df ignore terms that have a document frequency higher than threshold when building vocab
# ngram_range a tuple of lower and upp \er boundary of range of n-values for different n-grams to be extracted.
# extract 2-grams of words in addition to be 1-grams of individual words
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=False, stop_words='english')
tv_train = tv.fit_transform(trainX.ravel())
tv_test = tv.transform(testX.ravel())  # transform test review into features

print(f'tfidf model: train: {tv_test.shape} || test: {tv_test.shape}')
print('feature extraction --- end')

feature = tv.get_feature_names()[0:200]

print()

print('prediction --- start')
# predict
# multinomial naive bayes predict on tfidf
mnb = MultinomialNB()
mnb.fit(tv_train, trainY)  # build model
tv_pred = mnb.predict(tv_test)  # predict using model
print('prediction --- end')

print()

# evaluation
# reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
print('\nModel Evaluation:')
tv_accuracy = np.round(mt.accuracy_score(testY, tv_pred), 2)
tv_precision = np.round(mt.precision_score(testY, tv_pred, average='macro'), 2)
tv_recall = np.round(mt.recall_score(testY, tv_pred, average='macro'), 2)
tv_f1 = np.round(mt.f1_score(testY, tv_pred, average='macro'), 2)

tv_metrics = np.array([tv_accuracy, tv_precision, tv_recall, tv_f1])
tv_metrics = pd.DataFrame([tv_metrics], columns=['accuracy', 'precision', 'recall', 'f1'], index=['metrics'])
print('Performance Metrics:')
print(tabulate(tv_metrics, headers='keys', tablefmt='github'))

# visualization
fig = plt.figure()
ax = tv_metrics.plot.bar()
plt.title('MultinomialNB Performance Evaluation\n')
plt.ylabel('result')
plt.xlabel('model evualtion')
plt.xticks(rotation=-360)  # rotate x labels
plt.ylim([0.1, 1.0])
plt.show()

print('\nConfusion Matrix of MultinomialNB:\n')

# display and plot confusion matrix of bag-of-word
labels = ['positive', 'negative', 'neutral']
cv_cm = mt.confusion_matrix(testY, tv_pred, labels=labels)

# plot
# references: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785
# display and plot confusion matrix
tv_cm = mt.confusion_matrix(testY, tv_pred, labels=labels)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Confusion Matrix of MultinomialNB\n')
fig.colorbar(ax.matshow(tv_cm))
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('predicted')
plt.ylabel('true')
plt.show()

# display in table format
level = [len(labels)*[0], list(range(len(labels)))]
tv_cmf = pd.DataFrame(data=tv_cm,
                      columns=pd.MultiIndex(levels=[['predicted:'], labels], labels=level),
                      index=pd.MultiIndex(levels=[['actual:'], labels], labels=level))
print(tv_cmf)

print('\nClassification Report of MultinomialNB with TF-IDF Feature Extraction:\n')

# classification report for tf-idf
tv_report = mt.classification_report(testY, tv_pred, labels=labels)
print(tv_report)

print()

# end of ml sentiment analysis

"""
Appendix
"""
# # feature extraction
# # build bag of world feature and tfidf feature on train reviews
# # reference: https://stackoverflow.com/questions/18200052/how-to-convert-ndarray-to-array
# print('feature extraction --- start')
# cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1, 2))
# cv_train = cv.fit_transform(trainX.ravel())
# cv_test = cv.transform(testX.ravel())  # transform test review into features

# # multinomial naive bayes predict on bow
# mnb = MultinomialNB()
# mnb.fit(tv_train, trainY)  # build model
# tv_pred = mnb.predict(tv_test)  # predict using model
# print('prediction --- end')

# # prediction
# print('\nModel Evaluation with Bag of Word:')
# cv_accuracy = np.round(mt.accuracy_score(testY, cv_pred), 2)
# cv_precision = np.round(mt.precision_score(testY, cv_pred, average='macro'), 2)
# cv_recall = np.round(mt.recall_score(testY, cv_pred, average='macro'), 2)
# cv_f1 = np.round(mt.f1_score(testY, cv_pred, average='macro'), 2)
#
# cv_metrics = np.array([cv_accuracy, cv_precision, cv_recall, cv_f1])
# cv_metrics = pd.DataFrame([cv_metrics], columns=['accuracy', 'precision', 'recall', 'f1'], index=['bag of word'])
# print('Performance Metrics:')
# print(tabulate(cv_metrics, headers='keys', tablefmt='github'))

# # display and plot confusion matrix of bag-of-word
# labels = ['positive', 'negative', 'neutral']
# cv_cm = mt.confusion_matrix(testY, cv_pred, labels=labels)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.title('Confusion Matrix of MultinomialNB with Bag of Word Feature Extraction\n')
# fig.colorbar(ax.matshow(cv_cm))
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('predicted')
# plt.ylabel('true')
# plt.show()
#
# # display in table format
# level = [len(labels)*[0], list(range(len(labels)))]
# cv_cmf = pd.DataFrame(data=cv_cm,
#                       columns=pd.MultiIndex(levels=[['predicted:'], labels], labels=level),
#                       index=pd.MultiIndex(levels=[['actual:'], labels], labels=level))
# print(cv_cmf)

# print('\nConfusion Matrix of MultinomialNB with Bag of Word Feature Extraction:\n')
#
# # classification report for bag-of-word
# cv_report = mt.classification_report(testY, cv_pred, labels=labels)
# print(cv_report)
