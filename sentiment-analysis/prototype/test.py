import re
import nltk
import pandas as pd

from nltk import pos_tag
from lemmatization import PART
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Load CSV and read first 100 rows
path = '../datasets/reviews_of_amazon_products_datasets.csv'
df = pd.read_csv(path, index_col=None, na_values=['NA'], sep=',', low_memory=False, nrows=10)


# English Stop Words Removal: e.g. https://kb.yoast.com/kb/list-stop-words/
sw = set(stopwords.words('english'))

# Words Lemmatizing
lemmatizer = WordNetLemmatizer()

# Words Stemming
stemmer = SnowballStemmer('english')

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()


# Read Stop Words Minimal List
# References: http://text-analytics101.rxnlp.com/2014/10/all-about-stop-words-for-text-mining.html
minimal_sw = []
swm = open("../../minimal-stop.txt", "r")
pattern = re.compile(r'^\w+')
for line in swm:
    words = pattern.findall(line)
    for w in words:
        minimal_sw.append(w)
swm.close()


# References: https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
def preprocess(reviews):
    reviews = lowercase(reviews)
    reviews = add_white_space(reviews)
    reviews = expand_contractions(reviews)
    reviews = alpha_only(reviews)
    reviews = remove_punctuation(reviews)
    reviews = remove_whitespace(reviews)
    reviews = tokenize(reviews)
    reviews = tag_and_lem(reviews)
    return reviews


# Convert words to lowercase and add white space after a period. Some reviews does not have proper format.
def lowercase(word):
    return word.lower()


# Add white space after a period. Some reviews does not have proper format
def add_white_space(word):
    return re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', word))


# Remove numbers
def alpha_only(word):
    return re.sub(r'\w*\d\w*', '', word)


# Remove punctuation
def remove_punctuation(word):
    return re.sub(r'[^\w\s]', '', word)


# Expand Contractions - Reference: https://gist.github.com/dipanjanS/ae90598a0145b072926831a74f699ccd
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


# Remove whitespace
def remove_whitespace(word):
    return word.strip()


# Tokenizing
def tokenize(sentence):
    return word_tokenize(sentence)


# Lemmatization
# Reference: https://github.com/KT12/tag-lemmatize/blob/master/tag-lemmatize.py
def convert_tag(penn_tag):
    """
    convert_tags() accepts the first letter of a Penn POS Tag,
    then uses a dictionary to convert it to the appropriate WordNet tag.
    """
    if penn_tag in PART.keys():
        return PART[penn_tag]
    else:
        # other parts of speech will be tagged as nouns
        return 'n'


def tag_and_lem(element):
    tokens = pos_tag(element)
    return ' '.join([lemmatizer.lemmatize(tokens[k][0], convert_tag(tokens[k][1][0]))
                     for k in range(len(element))])


# Stop Words Removal
# Reference: https://www.quora.com/Stop-word-removal-or-POS-tagging-which-goes-first
# def remove_stopwords(tokens):
#     return filter(lambda token: token not in sw, tokens)
#
#
# # Stemming
# def stemming(tokens):
#     return [stemmer.stem(w) for w in tokens]
#
#
# # Shallow Parsing or Chunking
# def chunking(pos):
#     reg_exp = "NP: {<DT>?<JJ>*<NN>}"
#     regex = nltk.RegexpParser(reg_exp)
#     # visualize = regex.parse(pos).draw()
#     return regex.parse(pos)


df['filtered.reviews'] = df.apply(lambda row: preprocess(row['reviews.text']), axis=1)

# Get certain column
data = df[['asins', 'name', 'brand', 'filtered.reviews']]


# Sentiment Scoring
def sentiment_scores(sentence):
    print(analyzer.polarity_scores(sentence))


data.apply(lambda row: sentiment_scores(row['filtered.reviews']), axis=1)
