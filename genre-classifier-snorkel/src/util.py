from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()

files_to_tokens = {}

def normalize_token(token):
    return ps.stem(token).lower()

import pathlib
directory = pathlib.Path(__file__).parent.parent.resolve()

for f in ['discussion.txt']:
    with open(directory / 'resources' / 'wordlists' / f) as wordlist:
        files_to_tokens[f] = set(normalize_token(i) for i in wordlist.read().split('\n'))

for f in ['vocabulary-popescul-modified-discussion.txt', 'vocabulary-popescul-modified-download.txt', 'vocabulary-popescul-modified-articles.txt', 'vocabulary-popescul-modified-linklists.txt', 'vocabulary-popescul-modified-portrait-non_priv.txt', 'vocabulary-popescul-modified-portrait-priv.txt', 'vocabulary-popescul-modified-shop.txt']:
    with open(directory / 'resources' / 'vocabulary' / f) as wordlist:
        files_to_tokens[f] = set(i for i in wordlist.read().split())


def extract_overlapping_terms(text, filename):
    ret = []
    for term in word_tokenize(text):
        if normalize_token(term) in files_to_tokens[filename]:
            ret += [term]

    return ret