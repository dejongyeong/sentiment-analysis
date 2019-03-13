# Import Data
import pandas as pd
from utils import convert_tag, lexicon_sentiment, sentiment

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

# print(f'Output...')
pd.DataFrame(data).to_csv(r'../datasets/amazon_unlocked_mobile_datasets_with_sentiment.csv', index=False, header=None)
