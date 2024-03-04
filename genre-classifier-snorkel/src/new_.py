import spacy
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer


def get_tokens_info(doc):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(doc)
    tokens = []
    for token in doc:
        tokens.append((token.lemma_, token.pos_, token.tag_, token.is_alpha, token.is_stop))
    return [t for t in tokens if not t[4]]

def normalise_wordlists(word_list):
    ps = PorterStemmer()
    stem_list = []
    for w in word_list:
        stem_list.append(ps(w))

# possible solution 
# from test dataset zenodo create vocaluary lists for each lable 
# check is the key words in the document contained in any vocabulary 
# can also use the similarity from spacy doc1.similarity(doc2)
        
# todo
def create_voc(lable):
    pass
# todo
def check_similarity():
    pass
        







