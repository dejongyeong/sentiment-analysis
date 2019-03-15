"""
Created on Sun Feb 03 12:45am 2019
@author: De Jong Yeong (T00185309)

Utility function for sentiment analysis implementation.

Output file: prepared_amazon_unlocked_mobile_datasets.csv
"""

import numpy as np
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import sentiwordnet as swn


# part of speeh tagging and wordnet lemmatization
# convert penn treebank tag to wordnet tag
# reference: https://github.com/prateek22sri/Sentiment-analysis/blob/master/unigramSentiWordNet.py
# reference: https://github.com/KT12/tag-lemmatize/blob/master/tag-lemmatize.py
# reference: https://wordnet.princeton.edu/documentation/wnintro3wn
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


token = RegexpTokenizer(r'\w+')


# sentiwordnet sentiment scoring
# reference: https://sentiwordnet.isti.cnr.it/
# reference: https://www.tutorialspoint.com/How-to-catch-StopIteration-Exception-in-Python
# usage: sentiwordnet.senti_synsets('good', 'n')
def lexicon_sentiment(review):
    tagged = pos_tag(token.tokenize(review))
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


def sentiment(final_score):
    if final_score is None:
        return np.nan
    elif final_score > 0.0:
        return "positive"
    elif final_score < 0.0:
        return "negative"
    else:
        return "neutral"