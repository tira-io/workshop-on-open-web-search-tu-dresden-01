from collections import Counter
import process_labels
import spacy

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
from collections import Counter
import nltk
import json
import numpy as np

files_to_tokens = {}

def normalize_token(token):
    return ps.stem(token).lower()

stopwords = set(nltk.corpus.stopwords.words('english'))

import pathlib
directory = "/workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/"
for f in ['vocabulary-popescul-modified-discussion.txt', 'vocabulary-popescul-modified-download.txt', 'vocabulary-popescul-modified-articles.txt', 'vocabulary-popescul-modified-linklists.txt', 'vocabulary-popescul-modified-portrait-non_priv.txt', 'vocabulary-popescul-modified-portrait-priv.txt', 'vocabulary-popescul-modified-shop.txt', 'vocabulary-popescul-modified-help.txt']:
    with open(directory + '/resources/vocabulary/' + f, 'r') as wordlist:
        files_to_tokens[f] = set(i for i in wordlist.read().split())
#        files_to_tokens[f] = set(i for i in files_to_tokens[f] if i not in stopwords)

# Constants for the labels
ABSTAIN = -1
ARTICLES = 0
DISCUSSION = 1
DOWNLOAD = 2
HELP = 3
LINKLISTS = 4
PROTAIT_NPRIV = 5
PORTRAIT_PRIV = 6
SHOP = 7

# Constants for the labels
label_names = {DISCUSSION: 'Discussion', SHOP: 'Shop', ABSTAIN: 'Abstain', DOWNLOAD : 'Download', ARTICLES : 'Articles',
                HELP : 'Help', LINKLISTS : 'Linklists', PORTRAIT_PRIV : 'Porttrait private', PROTAIT_NPRIV : 'Protrait non private'}
def extract_overlapping_terms(processed_document, field, filename):
    ret = []
    for term in processed_document[field]:
        if term in files_to_tokens[filename]:
            ret += [term]

    return ret

def classifier_based_on_most_frequent_terms(doc):
    article_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_75', 'vocabulary-popescul-modified-articles.txt'))
    download_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_75', 'vocabulary-popescul-modified-download.txt'))
    discussion_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_75', 'vocabulary-popescul-modified-discussion.txt'))
    linklists_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_75', 'vocabulary-popescul-modified-linklists.txt'))
    portrait_non_priv_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_75', 'vocabulary-popescul-modified-portrait-non_priv.txt'))
    portrait_priv_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_75', 'vocabulary-popescul-modified-portrait-priv.txt'))
    shop_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_75', 'vocabulary-popescul-modified-shop.txt'))
    help_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-help.txt'))

    max_value = max(article_terms_len, download_terms_len, discussion_terms_len, linklists_terms_len, portrait_non_priv_terms_len, portrait_priv_terms_len, shop_terms_len)
    if (max_value == article_terms_len):
        return ARTICLES
    elif (max_value == download_terms_len):
        return DOWNLOAD
    elif (max_value == discussion_terms_len):
        return DISCUSSION
    elif (max_value == linklists_terms_len):
        return LINKLISTS
    elif (max_value == portrait_non_priv_terms_len):
        return PROTAIT_NPRIV
    elif (max_value == portrait_priv_terms_len):
        return PORTRAIT_PRIV
    elif (max_value == shop_terms_len):
        return SHOP
    elif(max_value == help_terms_len):
        return HELP
    else:
        return ABSTAIN

def classifier_based_on_most_frequent_terms_with_threshold(doc, offset=5):
    article_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-articles.txt'))
    download_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-download.txt'))
    discussion_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-discussion.txt'))
    linklists_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-linklists.txt'))
    portrait_non_priv_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-portrait-non_priv.txt'))
    portrait_priv_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-portrait-priv.txt'))
    shop_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-shop.txt'))
    help_terms_len = len(extract_overlapping_terms(doc, 'tokens_with_count_100', 'vocabulary-popescul-modified-help.txt'))

    nums = [article_terms_len, download_terms_len, discussion_terms_len, linklists_terms_len, portrait_non_priv_terms_len,
            portrait_priv_terms_len, shop_terms_len ]
    
    threshold = sorted(nums, reverse=True)[1] + offset
    if (threshold <= article_terms_len):
        return ARTICLES
    elif (threshold <= download_terms_len):
        return DOWNLOAD
    elif (threshold <= discussion_terms_len):
        return DISCUSSION
    elif (threshold <= linklists_terms_len):
        return LINKLISTS
    elif (threshold <= portrait_non_priv_terms_len):
        return PROTAIT_NPRIV
    elif (threshold <= portrait_priv_terms_len):
        return PORTRAIT_PRIV
    elif (threshold <= shop_terms_len):
        return SHOP
    elif (threshold <= help_terms_len):
        return HELP
    else:
        return ABSTAIN
    
def classifier_based_on_vector_space(doc):
    dict_tf = {}
    for json_file in ['0.json', '1.json', '2.json', '3.json', '4.json', '5.json', '6.json', '7.json']:
        with open(directory + 'resources/Json_stemmed_word/' + json_file) as data:
            dict_tf[int(json_file[0])] = json.load(data)
    with open(directory + 'resources/Json_stemmed_word/all_terms.json') as data:
        dict_df = json.load(data)

    dict_tf_idf = {}

    for key, value in dict_tf.items():
        # Initialize dict_tf_idf[key] if it doesn't exist
        if key not in dict_tf_idf:
            dict_tf_idf[key] = {}
        
        for word, tf in value.items():
            if word in dict_df:
                df = dict_df[word]
                dict_tf_idf[key][word] = tf / df

    for key, value in dict_tf_idf.items():
        # Sort the dictionary by value in descending order and take only the highest 100
        sorted_value = dict(sorted(value.items(), key=lambda item: item[1], reverse=True)[:200])
        dict_tf_idf[key] = sorted_value

    all_words = set(word for sub_dict in dict_tf_idf.values() for word in sub_dict.keys())

    # Create a dictionary to store numpy arrays for each word
    word_vectors = {word: np.zeros(len(dict_tf_idf)) for word in all_words}

    # Iterate through each sub-dictionary and fill in the values
    for key, sub_dict in dict_tf_idf.items():
        for word, value in sub_dict.items():
            word_vectors[word][key] = value

    # Convert the numpy arrays to lists if needed
    word_vectors = {word: vector.tolist() for word, vector in word_vectors.items()}
    #print(doc)
    #for t in word_tokenize(doc):
     #   if t not in stopwords:
      #      print(normalize_token(t))
        
    word_list_of_doc = [normalize_token(t) for t in word_tokenize(doc) if t not in stopwords]
    word_list_of_doc_within_corpus = [w for w in word_list_of_doc if w in word_vectors.keys()]
    word_vectors_of_doc = [np.array(word_vectors[word]) for word in word_list_of_doc_within_corpus]
    document_score = np.prod([word_vector + 1 for word_vector in word_vectors_of_doc], axis = 0)
    return np.argmax(document_score)

    

