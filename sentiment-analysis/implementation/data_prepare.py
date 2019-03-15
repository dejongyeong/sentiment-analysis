"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import re
import sys
import numpy as np
import pandas as pd
from time import sleep
from nltk import pos_tag
from langua import Predict
from utils import convert_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from contractions import CONTRACTION_MAP
from nltk.tokenize import RegexpTokenizer
from symspellpy.symspellpy import SymSpell

# create dictionary for spelling correction
# reference: https://github.com/mammothb/symspellpy
ss = SymSpell(2, 7)  # maximum edit distance per ditionary calculate, prefix length
path = '../datasets/frequency_dictionary_en_82_765.txt'
if not ss.create_dictionary(path):
    print('dictionary not found')
    sys.exit()

# load csv file with specific column only
filename = '../datasets/amazon_unlocked_mobile_datasets.csv'
fields = ['Product Name', 'Brand Name', 'Reviews']
data = pd.read_csv(filename, usecols=fields, nrows=100000)

"""
Data Understanding and Analyzing
"""
print(f'\nBefore Cleaning:')
shape = data.shape
types = data.dtypes
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Dimensions: {data.ndim}')
print(f'No. of Null Values: {data.isnull().sum().sum()}')

# clean data
# remove null value
review = data.dropna(axis=0, how='any')  # assign cleaned data into new variable.
shape = review.shape
print(f'\nAfter Cleaning:')
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Null Values: {review.isnull().sum().sum()}\n')

# strip spaces
review.columns = review.columns.str.strip()

# rename column header
print(f'Initial: {review.columns}')
review.columns = ['product.name', 'brand.name', 'review.text']
print(f'Renamed: {review.columns}\n\n')

"""
Data Preprocessing
"""
# turn off SettingWithCopy warning
pd.set_option('mode.chained_assignment', None)

# remove url
print('remove url --- start')
for index, row in review.iterrows():
    review.at[index, 'review.process'] = re.sub(r'https?\S+', '', str(row['review.text']))
    print(f'remove url... {index}')
print('remove url --- finish')

print()

# remove digits
# Reference: https://stackoverflow.com/questions/12851791/removing-numbers-from-string
print('remove digits --- start')
for index, row in review.iterrows():
    if len(str(row['review.process'])) < 1:
        continue
    else:
        review.at[index, 'review.process'] = re.sub(r'\+|\d+', '', str(row['review.process']))
    print(f'remove digits... {index}')
print('remove digits --- finish')

# predict and remove non-english sentence
# references: https://github.com/whiletruelearn/langua || https://pypi.org/project/langua/
pred = Predict()
print('remove non-english sentence --- start')
for index, row in review.iterrows():
    try:
        if len(str(row['review.process'])) >= 1:
            lang = pred.get_lang(str(row['review.process']))
            if lang != 'en':
                review.at[index, 'review.process'] = ''
    except:
        pass
    print(f'remove non-english sentence... {index}')
print('remove non-english sentence --- finish')

print()

# replace '' rows with nan and remove
# reason: only perform sentiment analysis on english text.
print('remove empty string rows --- start')
review['review.process'] = review['review.process'].apply(lambda x: np.nan if len(x) == 0 else x)
review = review[pd.notnull(review['review.process'])]
print('remove empty string rows --- finish')

print()

# insert space between punctuations and letters
# https://stackoverflow.com/questions/20705832/python-regex-inserting-a-space-between-punctuation-and-letters/20705997
print(f'insert space between punctuation and letter --- start')
for index, row in review.iterrows():
    review.at[index, 'review.process'] = re.sub(r'([a-zA-Z])([,.!()])', r'\1\2 ', str(row['review.process']))
    print(f'insert space... {index}')
print(f'insert space between punctuation and letter --- finish')


# expand contractions
# reference: https://gist.github.com/dipanjanS/ae90598a0145b072926831a74f699ccd
def expand_contractions(word, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

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


print(f'expand contractions --- start')
for index, row in review.iterrows():
    review.at[index, 'review.process'] = expand_contractions(str(row['review.process']))
    print(f'expand contraction... {index}')
print(f'expand contractions --- finish')

print()

# spelling correction
# reference: https://pypi.org/project/symspellpy/
print(f'spelling correction --- start')
for index, row in review.iterrows():
    suggestion = ss.lookup_compound(str(row['review.process']), 2)  # max edit distance per lookup
    for s in suggestion:
        review.at[index, 'review.process'] = s.term
    print(f'spelling correction... {index}')
print(f'spelling correction --- finish')

print()

# lowercase
print(f'lowercase --- start')
for index, row in review.iterrows():
    review.at[index, 'review.process'] = row['review.process'].lower()
    print(f'lowercase... {index}')
print(f'lowercase --- finish')

print()

# regex tokenizer remove punctuation and whitespaces
# reference: https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
print(f'regex tokenization --- start')
tokenizer = RegexpTokenizer(r'\w+')
review['review.tokened'] = review.apply(lambda item: tokenizer.tokenize(str(item['review.process'])),
                                        axis=1)
sleep(0.5)
print(f'tokenize...')
print(f'regex tokenization --- finish')

print()

# remove stopwords
# reference: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# reference: https://gist.github.com/sebleier/554280
print(f'remove stopwords --- start')
stopset = stopwords.words('english')
stopset = [item for item in stopset if item not in ('no', 'not', 'nor')]
for index, row in review.iterrows():
    filtered = [word for word in row['review.tokened'] if word not in stopset]
    filtered = []
    for word in row['review.tokened']:
        if word not in stopset:
            filtered.append(word)
    review.at[index, 'review.tokened'] = filtered
    print(f'remove stopwords... {index}')
print(f'remove stopwords --- finish')

print()


# lemmmatization
def tag_and_lemm(element):
    """
    accepts a tokenized sentences, tags, convert tags, lemmatizing
    """
    lemmatizer = WordNetLemmatizer()
    sentence = pos_tag(element)
    words = []

    # list of tuples [('token', 'tag'), ('token2', 'tag2')...]
    for word, tag in sentence:
        wn_tag = convert_tag(tag)
        if wn_tag is None:
            continue
        else:
            words.append(lemmatizer.lemmatize(word, wn_tag))

    return ' '.join(words)


print(f'lemmatization --- start')
for index, row in review.iterrows():
    review.at[index, 'review.tokened'] = tag_and_lemm(row['review.tokened'])
    print(f'lemmatize... {index}')
print(f'lemmatization --- finish')

print()

# output data with sentiment into new csv file
# reference: https://stackoverflow.com/questions/46098401/pandas-write-to-string-to-csv-instead-of-an-array
print(f'output --- start')
pd.DataFrame(review).to_csv(r'../datasets/prepared_amazon_unlocked_mobile_datasets.csv', index=False, header=None)
sleep(0.5)
print(f'output --- finish')

"""
Appendix
"""
# # remove punctuation and multiple whitespace
# # references: https://pythonadventures.wordpress.com/2017/02/05/remove-punctuations-from-a-text/
# print(f'Start remove punctuation...')
# for index, row in review.iterrows():
#     review.at[index, 'review.clean'] = str(row['review.clean']).translate(str.maketrans("", "", string.punctuation))
#     review.at[index, 'review.clean'] = re.sub(' +', ' ', row['review.clean'])
#     print(f'processing {index}...')
# print(f'End remove punctuation...\n')

# print()

# spelling correction
# # references: https://textblob.readthedocs.io/en/dev/quickstart.html#spelling-correction
# # cons: slow performance using textblob, longer time required for data analyzing.
# # references: https://norvig.com/spell-correct.html
# print(f'spelling correction --- start')
# for index, row in review.iterrows():
#     review.at[index, 'review.process'] = TextBlob(str(row['review.process'])).correct()
#     print(f'process... {index}')
# print(f'spelling correction --- finish')
