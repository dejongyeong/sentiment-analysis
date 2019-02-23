"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import re
import sys
import string
import numpy as np
import pandas as pd
from sklearn import metrics
from textblob import TextBlob
from translate import Translator
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from contractions import CONTRACTION_MAP
from nltk.corpus import sentiwordnet as swn
from sklearn.preprocessing import MultiLabelBinarizer

# Load CSV file with specific column only
filename = '../datasets/amazon_unlocked_mobile_datasets.csv'
fields = ['Product Name', 'Brand Name', 'Reviews']
data = pd.read_csv(filename, low_memory=False, usecols=fields)


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
# Remove numbers in String
# Reference: https://stackoverflow.com/questions/12851791/removing-numbers-from-string
print(f'Start removing numbers...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = ''.join(w for w in str(row['Reviews']) if not w.isdigit())
    print(f'processing...')
print(f'End removing numbers..\n')

# Convert Non-English word to English and Spelling Correction
# TextBlob translation and language detection - powered by Google Translate
# Note: 100% gooddd! is detected as Welsh by Google, and translated to 100% free!
# Reference: https://pypi.org/project/translate/
print(f'Start translation non-english to english...')
translator = Translator(to_lang='en')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = translator.translate(str(row['Reviews']))
    print('translating...')
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


print(f'Start expand contractions...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = expand_contractions(str(row['Reviews']))
    print(f'processing...')
print(f'End expand contractions...\n')


# Remove Punctuation - Efficient and Remove Multiple Whitespace
# References: https://pythonadventures.wordpress.com/2017/02/05/remove-punctuations-from-a-text/
print(f'Start remove punctuation...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = str(row['Reviews']).translate(str.maketrans("", "", string.punctuation))
    reviews.at[index, 'Reviews'] = re.sub(' +', ' ', str(row['Reviews']))
    print(f'processing...')
print(f'End remove punctuation...\n')

# Lowercase
reviews['Reviews'] = reviews['Reviews'].str.lower()

# Tokenization
print(f'Start tokenization...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = word_tokenize(str(row['Reviews']), language='english')
    print(f'tokenizing...')
print(f'End tokenization...\n')

# Stop Words Removal
# Python Lambda funtion in List Comprehension
# Reference: https://stackoverflow.com/questions/33245567/stopword-removal-with-nltk-and-pandas/33246035
print(f'Start stopwords removal...')
stopset = stopwords.words('english')
reviews['Reviews'] = reviews['Reviews'].apply(lambda x: [item for item in x if item not in stopset])
print(f'processing...')
print(f'End stopwords removal...\n')


# Part of Speeh Tagging and Lemmatization
# Convert Penn treebank tag to WordNet Tag
# WordNet Lemmatizer with NLTK Libraries
# Reference: https://github.com/KT12/tag-lemmatize/blob/master/tag-lemmatize.py
# Reference: https://linguistics.stackexchange.com/questions/6508/which-part-of-speech-are-s-and-r-in-wordnet
def convert_tag(penn_tag):
    """
    convert_tag() accepts the **first letter** of a Penn part-of-speech tag,
    then uses a dict lookup to convert it to the appropriate WordNet tag.
    """
    part = {
        'N': 'n',  # Noun
        'V': 'v',  # Verb
        'J': 'a',  # Adjective
        'S': 's',  # Adjective Satellite
        'R': 'r'  # Adverb
    }

    if penn_tag in part.keys():
        return part[penn_tag]
    else:
        # other parts of speech will be tagged as nouns
        return 'n'


def tag_and_lemm(element):
    """
    accepts a tokenized sentences, tags, convert tags, lemmatizing
    """
    lemmatizer = WordNetLemmatizer()
    sentence = pos_tag(element)
    # list of tuples [('token', 'tag'), ('token2', 'tag2')...]
    return [lemmatizer.lemmatize(sentence[k][0], convert_tag(sentence[k][1][0])) for k in range(len(sentence))]


print(f'Start lemmatization...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = tag_and_lemm(row['Reviews'])
    print(f'lemmatizing...')
print(f'End lemmatization...\n')

# Remove Single Character Word after Tokenization
print(f'Start removing single character...')
for index, row in reviews.iterrows():
    reviews.at[index, 'Reviews'] = [w for w in row['Reviews'] if len(w) > 1]
    print(f'removing...')
print(f'End single character removal...\n')


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
        if final_score > 0.0:
            return [final_score, 'positive']
        elif final_score == 0.0:
            return [final_score, 'neutral']
        else:
            return [final_score, 'negative']


print(f'SentiWordNet Sentiment Scoring...')
for index, row in reviews.iterrows():
    label_sentiment = lexicon_sentiment(row['Reviews'])
    data.at[index, 'Score'] = label_sentiment[0]
    data.at[index, 'Sentiment'] = label_sentiment[1]
    print(f'scoring...')
print(f'End SentiWordNet Sentiment Scoring...\n')

# print(f'SentiWordNet Prediction...')
# swn_prediction = [lexicon_sentiment(review) for review in reviews['Reviews']]
# binarizer = MultiLabelBinarizer().fit_transform(swn_prediction)
# print(f"Accuracy: {np.round(metrics.accuracy_score(y_true=binarizer, y_pred=binarizer), 2)}")

# Overwrite original arrays into a single sentences instead of an array of words.
# Purpose is to output cleaned data into a new CSV file
# Reference: https://stackoverflow.com/questions/46098401/pandas-write-to-string-to-csv-instead-of-an-array
# reviews['Reviews'] = reviews['Reviews'].apply(lambda x: ' '.join(map(str, x)))
