# import data
import bisect
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import sklearn.model_selection as ms
from nltk.tokenize import RegexpTokenizer
from utils import lexicon_sentiment, sentiment
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

# load dataset
filename = '../datasets/amazon_unlocked_mobile_datasets_with_sentiment.csv'
names = ['product.name', 'brand.name', 'review.text', 'review.clean', 'score', 'sentiment']
data = pd.read_csv(filename, names=names, usecols=['review.clean', 'sentiment'])

enc = LabelEncoder()
token = RegexpTokenizer(r'\w+')

# split data 70% train 30% test
array = data.values
X = array[:, 0:1]
Y = array[:, 1]
size = 0.3

# testX and trainX is review.clean
# testY and trainY is sentiment
trainX, testX, trainY, testY = ms.train_test_split(X, Y, test_size=size)

# predict
print('start prediction')
predict = [sentiment(lexicon_sentiment(str(review))) for review in testX]
print('end prediction')

testY = enc.fit_transform(testY)
predict = enc.transform(predict)

# enc_class = enc.classes_.tolist()
# bisect.insort_left(enc_class, '<unknown>')
# enc.classes_ = enc_class
#
# # references: https://scikit-learn.org/stable/data_transforms.html
# testY = enc.fit_transform(testY)
# predict = enc.fit_transform(predict)

# metrics
# accuracy = np.round(mt.accuracy_score(testY, predict), 2)


# print(f'SentiWordNet Prediction...')
# swn_prediction = [lexicon_sentiment(review) for review in reviews['Reviews']]
# binarizer = MultiLabelBinarizer().fit_transform(swn_prediction)
# print(f"Accuracy: {np.round(metrics.accuracy_score(y_true=binarizer, y_pred=binarizer), 2)}")

