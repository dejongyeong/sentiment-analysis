# Import Data
import pandas as pd
import numpy as np
from utils import convert_tag
from nltk import word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from sklearn.preprocessing import MultiLabelBinarizer

# Load Dataset
filename = '../datasets/prepared_amazon_unlocked_mobile_datasets.csv'
names = ['product.name', 'brand.name', 'review.text', 'review.clean']
data = pd.read_csv(filename, names=names)

# Drop null values
data.dropna(how='any', axis=0, inplace=True)
print(f'\nNull values: {data.isnull().sum().sum()}\n')   # expect: 0


# SentiWordNet Sentiment Scoring
# Reference: https://sentiwordnet.isti.cnr.it/
# Reference: https://www.tutorialspoint.com/How-to-catch-StopIteration-Exception-in-Python
# Usage: sentiwordnet.senti_synsets('good', 'n')
def lexicon_sentiment(review):
    tagged = pos_tag(review)
    pos_score = neg_score = token_count = obj_score = 0
    ss_set = [swn.senti_synsets(tagged[k][0], convert_tag(tagged[k][1][0])) for k in range(len(tagged))]
    if ss_set:
        for word in ss_set:
            # take the first senti-synsets of each word
            try:
                w = next(iter(word))

                pos_score += w.pos_score()
                neg_score += w.neg_score()
                obj_score += w.obj_score()
                token_count += 1
            except StopIteration:
                # ignore exception
                pass

        # aggregate final scores
        if float(pos_score - neg_score) == 0:
            final_score = round(float(pos_score - neg_score), 3)
        else:
            final_score = round(float(pos_score - neg_score) / token_count, 3)

        # return array of [final_score, 'sentiment']
        return final_score


print(f'SentiWordNet Sentiment Scoring...')
for index, row in data.iterrows():
    score = lexicon_sentiment(word_tokenize(row['review.clean']))
    if score is None:
        sentiment = np.NaN
    elif score > 0.0:
        sentiment = "positive"
    elif score < 0.0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    data.at[index, 'scores'] = score
    data.at[index, 'sentiment'] = sentiment
    print(f'scoring {index}...')
print(f'End SentiWordNet Sentiment Scoring...\n')

print(f'Output...')
pd.DataFrame(data).to_csv(r'../datasets/amazon_unlocked_mobile_datasets_with_sentiment.csv', index=False, header=None)

# print(f'SentiWordNet Prediction...')
# swn_prediction = [lexicon_sentiment(review) for review in reviews['Reviews']]
# binarizer = MultiLabelBinarizer().fit_transform(swn_prediction)
# print(f"Accuracy: {np.round(metrics.accuracy_score(y_true=binarizer, y_pred=binarizer), 2)}")
