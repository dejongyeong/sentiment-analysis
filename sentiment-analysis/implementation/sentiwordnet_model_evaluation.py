"""
Created on Wed March 06 3:45pm 2019
@author: De Jong Yeong (T00185309)

Predict sentiment label with SentiWordNet dictionary and evaluate its performance including accuracy, precision, recall,
f1 score, confusion matrix and classification report. Data split into 70% training and 30% validation/test.

SentiWordNet (lexicon-based) takes in reviews in which stops word are removed and are lemmatized.

Book References:
Title: Text Analysis with Python: A Practical Real-World Approach to Gaining Actionable Insights from Your Data
Author: Dipanjan Sarkar
Publisher: APress (2016)
ISBN: 978-1-4842-2387-1

Output file: none
"""

# import data
import numpy as np
import pandas as pd
import sklearn.metrics as mt
from tabulate import tabulate
from matplotlib import pyplot as plt
import sklearn.model_selection as ms
from utils import lexicon_sentiment, sentiment
from sklearn.preprocessing import LabelEncoder

# load dataset
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

# predict
print('sentiwordnet prediction --- start')
pred = []
for index, rev in enumerate(testX):
    pred.append(sentiment(lexicon_sentiment(str(rev))))
    print(f'predict... {index}')
print('sentiwordnet prediction --- finish')

# convert to numpy array
pred = np.array(pred)

# label encoder
# references: https://towardsdatascience.com/encoding-categorical-features-21a2651a065c
le = LabelEncoder()
pred_le = le.fit_transform([x for x in list(map(str, pred)) if x is not None])  # deal with null values
testY_le = le.fit_transform(testY)

print('\nModel Evaluation:')

# performance metrics
# references: https://stackoverflow.com/questions/18528533/pretty-printing-a-pandas-dataframe
# references: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# references: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
accuracy = np.round(mt.accuracy_score(testY_le, pred_le), 2)
precision = np.round(mt.precision_score(testY_le, pred_le, average='macro'), 2)
recall = np.round(mt.recall_score(testY_le, pred_le, average='macro'), 2)
f1 = np.round(mt.f1_score(testY_le, pred_le, average='macro'), 2)

metrics = np.array([accuracy, precision, recall, f1])
metrics = pd.DataFrame([metrics], columns=['accuracy', 'precision', 'recall', 'f1'], index=['result'])
print('Performance Metrics:')
print(tabulate(metrics, headers='keys', tablefmt='github'))

# visualize performance metrics
# reference: https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
fig = plt.figure()
ax = metrics.plot.bar()
plt.title('Lexicon-based Approach Performance Evaluation\n')
plt.ylabel('result')
plt.xlabel('metrics')
plt.xticks(rotation=-360)  # rotate x labels
plt.ylim([0.1, 1.0])
for item in ax.patches:  # show value on plot
    ax.annotate(np.round(item.get_height(), decimals=2), (item.get_x() + item.get_width() / 2., item.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()

print('\nConfusion Matrix:\n')

# display and plot confusion matrix
labels = ['positive', 'negative', 'neutral']
cm = mt.confusion_matrix(testY, pred, labels=labels)

# plot
# references: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/48018785
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Confusion Matrix of SentiWordNet\n')
fig.colorbar(ax.matshow(cm))
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('predicted')
plt.ylabel('true')
plt.show()

# display in table format
level = [len(labels)*[0], list(range(len(labels)))]
cmf = pd.DataFrame(data=cm,
                   columns=pd.MultiIndex(levels=[['predicted:'], labels], labels=level),
                   index=pd.MultiIndex(levels=[['actual:'], labels], labels=level))
print(cmf)

print('\nClassification Report:\n')

# classification report
# references: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
report = mt.classification_report(testY, pred, labels=labels)
print(report)


"""
Appendix
"""
# prediction
# predict = [sentiment(lexicon_sentiment(str(review))) for review in testX]

# prediction
# swn_prediction = [lexicon_sentiment(review) for review in reviews['Reviews']]
# binarizer = MultiLabelBinarizer().fit_transform(swn_prediction)
# print(f"Accuracy: {np.round(metrics.accuracy_score(y_true=binarizer, y_pred=binarizer), 2)}")
