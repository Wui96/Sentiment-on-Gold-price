import numpy as np
import math
import datetime
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

SEED = 1234

def remove_punct(x):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in x:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def preprocessing(x):
    lower = []
    wordnet_lemmatizer = WordNetLemmatizer()
    stripped = remove_punct(x)
    splits = stripped.split()
    for split in splits:
        if split != '' and len(split) >= 3:
            lower.append(wordnet_lemmatizer.lemmatize(split))              #lower case
    return ' '.join(lower)

def preprocessing():
    news['headline'] = news['headline'].apply(lambda x: preprocessing(x))
    return 