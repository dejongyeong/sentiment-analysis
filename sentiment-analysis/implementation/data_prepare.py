"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)
"""

# Import Statement
import re
import string
import pandas as pd
from textblob import TextBlob
from translate import Translator
from nltk.corpus import stopwords, wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from contractions import CONTRACTION_MAP

# Load CSV file with specific column only
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

# Data Cleaning
# Remove Null Values
review = data.dropna(axis=0, how='any')  # assign cleaned data into new variable.
shape = review.shape
print(f'\nAfter Cleaning:')
print(f'No. of Rows: {shape[0]}\nNo. of Columns: {shape[1]}')
print(f'No. of Null Values: {review.isnull().sum().sum()}\n')

# Remove leading or trailing spaces
review.columns = review.columns.str.strip()

# Rename Columns Header
print(f'Initial: {review.columns}')
review.columns = ['product.name', 'brand.name', 'review.text']
print(f'Renamed: {review.columns}\n\n')


"""
Data Preprocessing
"""
# Remove numbers in String
# Reference: https://stackoverflow.com/questions/12851791/removing-numbers-from-string
print(f'Remove numbers and plus sign in review.text\n')
review['review.clean'] = review['review.text'].str.replace('\+|\d+', '')

# Convert Non-English word to English
# Reference: https://pypi.org/project/translate/
print(f'Translation non-english to english...')
translator = Translator(to_lang='en')
for index, row in review.iterrows():
    review.at[index, 'review.clean'] = translator.translate(str(row['review.clean']))
    print(f'translating {index}...')
print(f'End translation...\n')

# Regex Insert Space between Punctuation and Letters
# https://stackoverflow.com/questions/20705832/python-regex-inserting-a-space-between-punctuation-and-letters/20705997
print(f'Start inserting space...')
for index, row in review.iterrows():
    review.at[index, 'review.clean'] = re.sub(r'([a-zA-Z])([,.!()])', r'\1\2 ', str(row['review.clean']))
    print(f'processing {index}...')
print(f'End inserting space...\n')


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
for index, row in review.iterrows():
    review.at[index, 'review.clean'] = expand_contractions(str(row['review.clean']))
    print(f'processing {index}...')
print(f'End expand contractions...\n')

# Spelling Corrections - only for English word
# Install TextBlob and Download necessary NLTK corpora
# References: https://textblob.readthedocs.io/en/dev/quickstart.html#spelling-correction
# Cons: Slow performance using TextBlob, longer period for data analyzing.
# References: https://norvig.com/spell-correct.html
print(f'Start spelling corrections...')
for index, row in review.iterrows():
    review.at[index, 'review.clean'] = TextBlob(str(row['review.clean'])).correct()
    print(f'correcting {index}...')
print(f'End spelling corrections...\n')

# Remove Punctuation - Efficient and Remove Multiple Whitespace
# References: https://pythonadventures.wordpress.com/2017/02/05/remove-punctuations-from-a-text/
print(f'Start remove punctuation...')
for index, row in review.iterrows():
    review.at[index, 'review.clean'] = str(row['review.clean']).translate(str.maketrans("", "", string.punctuation))
    review.at[index, 'review.clean'] = re.sub(' +', ' ', row['review.clean'])
    print(f'processing {index}...')
print(f'End remove punctuation...\n')

# Lowercase
print(f'Start lowercasing...')
for index, row in review.iterrows():
    review.at[index, 'review.clean'] = row['review.clean'].lower()
    print(f'lowercasing {index}...')
print(f'End lowercasing...\n')

# Tokenization
print(f'Start tokenization...')
print(f'processing...')
review['review.token'] = review.apply(lambda row: word_tokenize(row['review.clean'], language='english'), axis=1)
print(f'End tokenization...\n')

# Stop Words Removal
# Reference: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# References: https://gist.github.com/sebleier/554280
print(f'Start stopwords removal...')
stopset = stopwords.words('english')
stopset = [item for item in stopset if item not in ('no', 'not', 'nor')]
for index, row in review.iterrows():
    filtered = [word for word in row['review.token'] if word not in stopset]
    filtered = []
    for word in row['review.token']:
        if word not in stopset:
            filtered.append(word)
    review.at[index, 'review.token'] = filtered
    print(f'processing {index}...')
print(f'End stopwords removal...\n')


# Part of Speeh Tagging and WordNet Lemmatization
# Convert Penn treebank tag to WordNet Tag
# Reference: https://github.com/prateek22sri/Sentiment-analysis/blob/master/unigramSentiWordNet.py
# Reference: https://github.com/KT12/tag-lemmatize/blob/master/tag-lemmatize.py
# Reference: https://wordnet.princeton.edu/documentation/wnintro3wn
def convert_tag(penn_tag):
    """
    Convert between PennTreebank to WordNet tags
    """
    if penn_tag.startswith('N'):     # Noun
        return wordnet.NOUN
    elif penn_tag.startswith('V'):   # Verb
        return wordnet.VERB
    elif penn_tag.startswith('J'):   # Adjective
        return wordnet.ADJ
    elif penn_tag.startswith('S'):   # Adjective Satellite
        return 's'
    elif penn_tag.startswith('R'):   # Adverb
        return wordnet.ADV
    else:
        return None  # other parts of speech will be returned as none


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

    return words


print(f'Start lemmatization...')
for index, row in review.iterrows():
    review.at[index, 'review.token'] = tag_and_lemm(row['review.token'])
    print(f'lemmatizing {index}...')
print(f'End lemmatization...\n')

# Purpose is to output data with sentiment into a new CSV file
# Reference: https://stackoverflow.com/questions/46098401/pandas-write-to-string-to-csv-instead-of-an-array
print(f'Output...')
pd.DataFrame(review).to_csv(r'../datasets/prepared_amazon_unlocked_mobile_datasets.csv', index=False, header=None)
