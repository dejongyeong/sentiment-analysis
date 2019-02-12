"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import re
import pandas as pd
from textblob import TextBlob

# Load CSV file with specific column only
filename = '../datasets/amazon_unlocked_mobile_datasets.csv'
fields = ['Product Name', 'Brand Name', 'Reviews']
data = pd.read_csv(filename, low_memory=False, usecols=fields, nrows=50)

# Data Understanding and Analyzing
shape = data.shape
types = data.dtypes
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Dimensions: {data.ndim}')
print(f'No. of Null Values: {data.isnull().sum().sum()}')

# Remove Null Values
data = data.dropna(axis=0, how='any')
shape = data.shape
print(f'\nAfter Cleaning:')
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Null Values: {data.isnull().sum().sum()}\n\n')

# Data Cleaning
# Remove Trailing Spaces
data.columns = data.columns.str.strip()

# Spelling Corrections
# Install TextBlob and Download necessary NLTK corpora
# References: https://textblob.readthedocs.io/en/dev/install.html
print(f'Start Spelling Corrections...')
for index, row in data.iterrows():
    data.at[index, 'Reviews'] = TextBlob(row['Reviews']).correct()
    print(f'processing...')
print(f'End Spelling Corrections...\n')

# Regex Insert Space between Punctuation and Letters References:
# https://stackoverflow.com/questions/20705832/python-regex-inserting-a-space-between-punctuation-and-letters/20705997
print(f'Start Insert Space between Punctuations and Letters...')
for index, row in data.iterrows():
    data.at[index, 'Reviews'] = re.sub(r'([a-zA-Z])([,.!()])', r'\1\2 ', str(row['Reviews']))
    print(f'processing...')
print(f'End Insert Space between Punctuations and Letters...\n')