"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import re
import sys
import string
import pandas as pd
from textblob import TextBlob
from contractions import CONTRACTION_MAP

# Load CSV file with specific column only
filename = '../datasets/amazon_unlocked_mobile_datasets.csv'
fields = ['Product Name', 'Brand Name', 'Reviews']
data = pd.read_csv(filename, low_memory=False, usecols=fields, nrows=76)


"""
Data Understanding and Analyzing
"""
shape = data.shape
types = data.dtypes
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Dimensions: {data.ndim}')
print(f'No. of Null Values: {data.isnull().sum().sum()}')

# Data Cleaning
# Remove Null Values
reviews = data.dropna(axis=0, how='any')
shape = reviews.shape
print(f'\nAfter Cleaning:')
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Null Values: {reviews.isnull().sum().sum()}\n\n')

# Remove Trailing Spaces
reviews.columns = reviews.columns.str.strip()


"""
Data Preprocessing
"""
# Convert Non-English word to English and Spelling Correction
# TextBlob translation and language detection - powered by Google Translate
# Note: 100% gooddd! is detected as Welsh by Google, and translated to 100% free!
print(f'Starting translation non-english to english...')
for index, row in reviews.iterrows():
    lang = TextBlob(str(row['Reviews'])).detect_language()
    if lang != 'en':
        initial = row['Reviews']
        reviews.at[index, 'Reviews'] = TextBlob(str(row['Reviews'])).translate(to='en')
        print(f"translating... {initial} -> {row['Reviews']}")
    print(f'skipping...')
print(f'End translation...\n')

# Spelling Corrections - only for English word
# Install TextBlob and Download necessary NLTK corpora
# References: https://textblob.readthedocs.io/en/dev/install.html
print(f'Start spelling corrections...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = TextBlob(str(row['Reviews'])).correct()
    print(f'correcting...')
print(f'End spelling corrections...\n')

# Regex Insert Space between Punctuation and Letters
# Remove Punctuation for better Sentiment Analysis
# https://stackoverflow.com/questions/20705832/python-regex-inserting-a-space-between-punctuation-and-letters/20705997
print(f'Starting insert space...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = re.sub(r'([a-zA-Z])([,.!()])', r'\1\2 ', str(row['Reviews']))
    print(f'processing...')
print(f'End insert space...\n')


# Expand Contractions
# Reference: https://gist.github.com/dipanjanS/ae90598a0145b072926831a74f699ccd
def expand_contractions(word, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expended_text = contractions_pattern.sub(expand_match, word)
    expended_text = re.sub("'", "", expended_text)
    return expended_text


print(f'Starting expand contractions...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = expand_contractions(str(row['Reviews']))
    print(f'processing...')
print(f'End expand contractions...\n')


# Remove Punctuation - Efficient and Remove Multiple Whitespace
# References: https://pythonadventures.wordpress.com/2017/02/05/remove-punctuations-from-a-text/
print(f'Starting remove punctuation...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = str(row['Reviews']).translate(str.maketrans("", "", string.punctuation))
    reviews.at[index, 'Reviews'] = re.sub(' +', ' ', str(row['Reviews']))
    print(f'processing...')
print(f'End remove punctuation...')

# Lowercase
reviews['Reviews'] = reviews['Reviews'].str.lower()

# TODO: Stop Word Removal
# TODO: Issue Detected - Spelling Correction: samsung -> samson; android -> andros
