import spacy
from nltk.stem import PorterStemmer
from tira.third_party_integrations import ir_datasets
from collections import Counter


def get_tokens_info(doc):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(doc)
    tokens = []
    for token in doc:
        tokens.append((token.lemma_, token.is_alpha, token.is_stop))
    return [t for t in tokens if t[1] and not t[2]]

def stemming_wordlists(word_list):
    ps = PorterStemmer()
    stem_list = []
    for w in word_list:
        stem_list.append(ps(w))

def tokens_with_count(tokens):
    tokens_with_count = Counter(tokens)
    return tokens_with_count

# possible solution 
# from test dataset zenodo create vocaluary lists for each lable 
# check is the key words in the document contained in any vocabulary 
# can also use the similarity from spacy doc1.similarity(doc2)
        
# todo
def create_voc(lable):
    pass

# todo
def get_similarity(doc, label):
    pass

def parse_doc(doc):
    new_ = {x[0]:0 for x in doc}
    for x in doc:
        new_[x[0]] += doc[x]
    return new_

def get_intersection_count(doc, label):
    intetsection = 0
    for token in doc:
        if token in label:
            intetsection += doc[token]
    return intersection

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Genre classifier using Snorkel')
    parser.add_argument('--input', type=str, default='workshop-on-open-web-search/document-processing-20231027-training', help='Dataset id from ir_datasets/TIRA to process.')
    
    return parser.parse_args()


if __name__ == '__main__':
    dataset = ir_datasets.load(parse_args().input)

    dataset_info = []
    for doc in dataset.docs_iter():
        doc = get_tokens_info(str(doc))
        doc = tokens_with_count(doc)
        doc = parse_doc(doc)
        dataset_info.append(doc)
    print(dataset_info)
        







