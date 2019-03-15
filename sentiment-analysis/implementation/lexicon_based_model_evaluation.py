"""
Created on Wed March 06 3:45pm 2019
@author: De Jong Yeong (T00185309)
"""

# # import data
# import numpy as np
# import pandas as pd
# import sklearn.metrics as mt
# import sklearn.model_selection as ms
# from nltk.tokenize import RegexpTokenizer
# from utils import lexicon_sentiment, sentiment
# from sklearn.preprocessing import MultiLabelBinarizer
#
# # load dataset
# filename = '../datasets/amazon_unlocked_mobile_datasets_with_sentiment.csv'
# names = ['product.name', 'brand.name', 'review.text', 'review.clean', 'score', 'sentiment']
# data = pd.read_csv(filename, names=names, usecols=['review.clean', 'sentiment'])
#
# token = RegexpTokenizer(r'\w+')
#
# # split data 70% train 30% test
# array = data.values
# X = array[:, 0:1]
# Y = array[:, 1]
# size = 0.4
#
# # testX and trainX is review.clean
# # testY and trainY is sentiment
# trainX, testX, trainY, testY = ms.train_test_split(X, Y, test_size=size, shuffle=True)
#
# # predict
# print('start prediction')
# predict = [sentiment(lexicon_sentiment(str(review))) for review in testX]
# print('end prediction')
#
# # https://stackoverflow.com/questions/51335535/multilabelbinarizer-output-classes-in-letters-instead-of-categories
# mlb = MultiLabelBinarizer()
# predict = mlb.fit_transform([x for x in list(map(str, predict)) if x is not None])  # deal with null values
# testY = mlb.fit_transform(testY)
#
# # metrics
# accuracy = np.round(mt.accuracy_score(testY, predict), 2)
# print(accuracy)
#
# # print(f'SentiWordNet Prediction...')
# # swn_prediction = [lexicon_sentiment(review) for review in reviews['Reviews']]
# # binarizer = MultiLabelBinarizer().fit_transform(swn_prediction)
# # print(f"Accuracy: {np.round(metrics.accuracy_score(y_true=binarizer, y_pred=binarizer), 2)}")


# print(f'SentiWordNet Sentiment Scoring...')
# for index, row in data.iterrows():
#     score = lexicon_sentiment(row['review.clean'])
#     data.at[index, 'scores'] = score
#     data.at[index, 'sentiment'] = sentiment(score)
#     print(f'scoring {index}...')
# print(f'End SentiWordNet Sentiment Scoring...\n')

