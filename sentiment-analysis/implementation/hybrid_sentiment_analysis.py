"""
Created on Monday April 1 2019 4:45pm
@author: De Jong Yeong (T00185309)

Hybrid approach for sentiment analysis. It is the combination of lexicon-based technique and machine learning technique
to perform sentiment analysis.

Reference: https://www.youtube.com/watch?v=LWJSc7JDs0s&t=12s
Reference: https://github.com/chrisjmccormick/LSA_Classification/blob/master/inspect_LSA.py

Output file: -

Future Work: consider emoticons like :D or :( when performing sentiment analysis.
"""

# import module
import numpy as np
import pandas as pd
import sklearn.metrics as mt
from tabulate import tabulate
from sklearn.svm import LinearSVC
import sklearn.model_selection as ms
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
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
print('feature extraction --- start')
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=False,
                     max_features=10000, smooth_idf=True, stop_words='english')
tv_train = tv.fit_transform(trainX.ravel())
tv_test = tv.transform(testX.ravel())  # transform test review into features
print('feature extraction --- end')

print()

# latent semantic analysis on train data (lexicon based)
print('latent semantic analysis --- start')
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))
lsa_train = lsa.fit_transform(tv_train)
lsa_test = lsa.transform(tv_test)  # transform tfidf test
print('latent semantic analysis --- end')

print()

# build model and predict with lsa train (machine learning)
print('build and predict --- start')
svm = LinearSVC()
svm.fit(lsa_train, trainY)
svm_pred = svm.predict(lsa_test)
print('build and predict --- end')

print()

# evaluation
print('\nModel Evaluation:')
tv_accuracy = np.round(mt.accuracy_score(testY, svm_pred), 3)
tv_precision = np.round(mt.precision_score(testY, svm_pred, average='macro'), 3)
tv_recall = np.round(mt.recall_score(testY, svm_pred, average='macro'), 3)
tv_f1 = np.round(mt.f1_score(testY, svm_pred, average='macro'), 3)

tv_metrics = np.array([tv_accuracy, tv_precision, tv_recall, tv_f1])
tv_metrics = pd.DataFrame([tv_metrics], columns=['accuracy', 'precision', 'recall', 'f1'], index=['metrics'])
print('Performance Metrics:')
print(tabulate(tv_metrics, headers='keys', tablefmt='github'))

# visualization
# reference: https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
fig = plt.figure()
ax = tv_metrics.plot.bar()
plt.title('Hybrid Approach Performance Evaluation\n')
plt.ylabel('result')
plt.xlabel('model evualtion')
plt.xticks(rotation=-360)  # rotate x labels
plt.ylim([0.1, 1.0])
for item in ax.patches:  # show value on plot
    ax.annotate(np.round(item.get_height(), decimals=2), (item.get_x() + item.get_width() / 2., item.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()
plt.savefig('../results/hybrid_performance.png')  # save result

print('\nConfusion Matrix of Hybrid Approach:\n')

# display and plot confusion matrix
labels = ['positive', 'negative', 'neutral']
svm_cm = mt.confusion_matrix(testY, svm_pred, labels=labels)

# plot
# display and plot confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Confusion Matrix of Hybrid Approach\n')
fig.colorbar(ax.matshow(svm_cm))
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('predicted')
plt.ylabel('true')
plt.show()
plt.savefig('../results/hybrid_confusion_matrix.png')  # save result

# display in table format
level = [len(labels)*[0], list(range(len(labels)))]
svm_cmf = pd.DataFrame(data=svm_cm,
                       columns=pd.MultiIndex(levels=[['predicted:'], labels], labels=level),
                       index=pd.MultiIndex(levels=[['actual:'], labels], labels=level))
print(svm_cmf)

# classification report for hybrid approach
svm_report = mt.classification_report(testY, svm_pred, labels=labels)
print(svm_report)

print()

# end of hybrid approach
