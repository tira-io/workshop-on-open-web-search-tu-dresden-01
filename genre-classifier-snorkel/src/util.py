from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
from collections import Counter
import nltk
#nltk.download('stopwords')

files_to_tokens = {}

def normalize_token(token):
    return ps.stem(token).lower()

stopwords = set(nltk.corpus.stopwords.words('english'))

import pathlib
directory = pathlib.Path(__file__).parent.parent.resolve()

for f in ['discussion.txt']:
    with open(directory / 'resources' / 'wordlists' / f) as wordlist:
        files_to_tokens[f] = set(normalize_token(i) for i in wordlist.read().split('\n'))

for f in ['vocabulary-popescul-modified-discussion.txt', 'vocabulary-popescul-modified-download.txt', 'vocabulary-popescul-modified-articles.txt', 'vocabulary-popescul-modified-linklists.txt', 'vocabulary-popescul-modified-portrait-non_priv.txt', 'vocabulary-popescul-modified-portrait-priv.txt', 'vocabulary-popescul-modified-shop.txt']:
    with open(directory / 'resources' / 'vocabulary' / f) as wordlist:
        files_to_tokens[f] = set(i for i in wordlist.read().split())
        files_to_tokens[f] = set(i for i in files_to_tokens[f] if i not in stopwords)
    

def preprocess_document(text, num=75):
    normalized_tokens = [normalize_token(t) for t in word_tokenize(text) if t not in stopwords]
    tokens_with_count = [i[0] for i in Counter(normalized_tokens).most_common(num)]
    return {
        'normalized_tokens': normalized_tokens,
        'normalized_token_set': list(set(normalized_tokens)),
        'tokens_with_count': tokens_with_count,
    }

def extract_overlapping_terms(processed_document, filename):
    ret = []
    for term in processed_document['tokens_with_count']:
        if term in files_to_tokens[filename]:
            ret += [term]

    return ret