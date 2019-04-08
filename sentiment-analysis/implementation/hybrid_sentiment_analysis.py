"""
Created on Monday April 1 2019 4:45pm
@author: De Jong Yeong (T00185309)

Hybrid approach for sentiment analysis. It is the combination of lexicon-based technique and machine learning technique
to perform sentiment analysis.

Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5912726/
Reference: https://www.sciencedirect.com/science/article/pii/S095741741630584X
Reference: https://www.irjet.net/archives/V3/i6/IRJET-V3I6539.pdf

Output file: -
"""

# import module
import re
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import sklearn.model_selection as ms
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from utils import lexicon_sentiment, sentiment
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

print('feature extraction --- start')
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=False)
tv_train = tv.fit_transform(trainX.ravel())
tv_test = tv.transform(testX.ravel())  # transform test review into features
print('feature extraction --- end')

print()