# import data
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.preprocessing import MultiLabelBinarizer

# load dataset
filename = '../datasets/amazon_unlocked_mobile_datasets_with_sentiment.csv'
names = ['product.name', 'brand.name', 'review.text', 'review.clean', 'score', 'sentiment']
fields = ['review.clean', 'sentiment']
data = pd.read_csv(filename, names=names, usecols=fields)

# print(f'SentiWordNet Prediction...')
# swn_prediction = [lexicon_sentiment(review) for review in reviews['Reviews']]
# binarizer = MultiLabelBinarizer().fit_transform(swn_prediction)
# print(f"Accuracy: {np.round(metrics.accuracy_score(y_true=binarizer, y_pred=binarizer), 2)}")

