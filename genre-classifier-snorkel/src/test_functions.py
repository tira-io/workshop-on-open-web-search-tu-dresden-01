from collections import Counter
import process_labels
import spacy
import util
from snorkel_genre_classifier import ABSTAIN, ARTICLES, DOWNLOAD, DISCUSSION, SHOP, PORTRAIT_PRIV, PROTAIT_NPRIV, LINKLISTS
nlp = spacy.load("en_core_web_sm")

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



def classifier_based_on_most_frequent_terms(doc, num = 100):
    most_frequent_terms = " ".join(get_most_frequent_terms(doc, num))
    article_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-articles.txt'))
    download_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-download.txt'))
    discussion_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-discussion.txt'))
    linklists_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-linklists.txt'))
    portrait_non_priv_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-portrait-non_priv.txt'))
    portrait_priv_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-portrait-priv.txt'))
    shop_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-shop.txt'))

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



def classifier_based_on_most_frequent_terms_with_threshold(doc, num = 100, offset=5):
    most_frequent_terms = " ".join(get_most_frequent_terms(doc, num))
    article_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-articles.txt'))
    download_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-download.txt'))
    discussion_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-discussion.txt'))
    linklists_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-linklists.txt'))
    portrait_non_priv_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-portrait-non_priv.txt'))
    portrait_priv_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-portrait-priv.txt'))
    shop_terms_len = len(util.extract_overlapping_terms(most_frequent_terms, 'vocabulary-popescul-modified-shop.txt'))

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