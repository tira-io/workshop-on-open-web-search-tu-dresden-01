from collections import Counter
import process_labels
import spacy
import util

nlp = spacy.load("en_core_web_sm")

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

def tokens_with_count(tokens):
    tokens_with_count = Counter(tokens)
    return tokens_with_count

def get_most_frequent_terms(doc, num=100):
    tokens = process_labels.get_tokens_info(doc, nlp)
    most_frequent_terms_with_count = tokens_with_count(tokens).most_common(num)
    most_frequent_terms = []
    for terms_with_count in most_frequent_terms_with_count:
        most_frequent_terms.append(terms_with_count[0][0])
    return most_frequent_terms

def read_token_of_each_class(path):
    with open(path, 'r') as file:
        content = file.readlines()
        content = [line.strip() for line in content]
    return content

'''
Classify documents by analyzing the top 75 tokens with the highest probability within each document. 
These tokens will be cross-referenced with the vocabulary in a designated vocabulary corpus. 
The document will then be categorized based on the classes associated with these tokens.
''' 

def classifier_based_on_most_frequent_terms(doc, dir_name='vocabulary-popescul-modified'):
    article_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_75', dir_name + '-articles.txt'))
    download_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_75', dir_name +'-download.txt'))
    discussion_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_75', dir_name +'-discussion.txt'))
    linklists_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_75', dir_name +'-linklists.txt'))
    portrait_non_priv_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_75', dir_name +'-portrait-non_priv.txt'))
    portrait_priv_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_75', dir_name +'-portrait-priv.txt'))
    shop_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_75', dir_name +'-shop.txt'))

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
    else:
        return ABSTAIN
    
'''
Classify documents by analyzing the top 75 tokens with the highest probability within each document. 
These tokens will be cross-referenced with the vocabulary in a designated vocabulary corpus. 
The document will then be categorized based on the classes associated with these tokens.
If  receiving uncertain result, documents will not be categorized (categorized as ABSTAIN)
''' 

def classifier_based_on_most_frequent_terms_with_threshold(doc, dir_name='vocabulary-popescul-modified', offset=5):
    article_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_100', dir_name +'-articles.txt'))
    download_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_100', dir_name +'-download.txt'))
    discussion_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_100', dir_name +'-discussion.txt'))
    linklists_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_100', dir_name +'-linklists.txt'))
    portrait_non_priv_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_100', dir_name +'-portrait-non_priv.txt'))
    portrait_priv_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_100', dir_name +'-portrait-priv.txt'))
    shop_terms_len = len(util.extract_overlapping_terms(doc, 'tokens_with_count_100', dir_name +'-shop.txt'))

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
    else:
        return ABSTAIN