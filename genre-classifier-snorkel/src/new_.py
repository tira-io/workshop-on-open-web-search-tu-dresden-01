import spacy
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer


def get_tokens_info(doc):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(doc)
    tokens = []
    for i in range(len(doc)):
        tokens.append((doc[i].text, doc[i].pos_, doc[i].tag_, doc[i].is_alpha, doc[i].is_stop, 
                       doc.ents[i].label_))
    print(tokens)
    return [t for t in tokens if not t[4]]

def normalise_wordlists(word_list):
    ps = PorterStemmer()
    stem_list = []
    for w in word_list:
        stem_list.append(ps(w))


        







