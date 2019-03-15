"""
Created on Tue Feb 26 4:45pm 2019
@author: De Jong Yeong (T00185309)
"""

# Import Data
import pandas as pd
from utils import lexicon_sentiment, sentiment

# Load Dataset
filename = '../datasets/prepared_amazon_unlocked_mobile_datasets.csv'
names = ['product.name', 'brand.name', 'review.text', 'review.clean']
data = pd.read_csv(filename, names=names)

# Drop null values
data.dropna(how='any', axis=0, inplace=True)
print(f'\nNull values: {data.isnull().sum().sum()}\n')   # expect: 0

print(f'SentiWordNet Sentiment Scoring...')
for index, row in data.iterrows():
    score = lexicon_sentiment(row['review.clean'])
    data.at[index, 'scores'] = score
    data.at[index, 'sentiment'] = sentiment(score)
    print(f'scoring {index}...')
print(f'End SentiWordNet Sentiment Scoring...\n')

print(f'Output...')
filename = r'../datasets/test_amazon_unlocked_mobile_datasets_with_sentiment.csv'
pd.DataFrame(data).to_csv(filename, index=False, header=None)
