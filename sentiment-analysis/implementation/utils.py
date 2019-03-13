from nltk.corpus import wordnet


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