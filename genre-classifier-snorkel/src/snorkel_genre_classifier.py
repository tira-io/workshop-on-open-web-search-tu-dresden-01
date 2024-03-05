#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import ir_datasets, get_output_directory
from pathlib import Path
import pandas as pd
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from nltk.stem import PorterStemmer 


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

directory = "/workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/resources/vocabulary/"
ps = PorterStemmer()


def lf_text_contains_discussion_term(doc):
    # TODO, use real word lists, maybe with stemming, stoppword removal, tokenization, etc.
    if 'discussion' in doc['text'] or 'discuss' in doc['text']:
        return DISCUSSION
    return ABSTAIN

def lf_text_contains_shop_term(doc):
    # TODO, use real word lists, maybe with stemming, stoppword removal, tokenization, etc.
    if 'quantity' in doc['text'] or 'buy' in doc['text']:
        return SHOP
    return ABSTAIN

# TODO ADD more and reasonable labeling functions
# Tutorial: https://www.snorkel.org/get-started/

def get_snorkel_pandas_lf_applier():
    lfs = [lf_text_contains_discussion_term, lf_text_contains_shop_term,
           # TODO, more labeling functions
          ]

    return PandasLFApplier(lfs=[LabelingFunction(name=func.__name__, f=func) for func in lfs])


def process_documents(document_iter):
    snorkel_applier = get_snorkel_pandas_lf_applier()
    df = pd.DataFrame([{'docno': i.doc_id, 'text': i.default_text()} for i in document_iter])
    document_features = snorkel_applier.apply(df)
    #todo, also test and maybe implement more advanced models
    label_model = MajorityLabelVoter(cardinality=len(label_names))

    predictions = label_model.predict(document_features)
    ret = []
    for i in range(len(document_features)):
        ret.append({'docno': df.iloc[i]['docno'], 'label': label_names[predictions[i]]})
    return pd.DataFrame(ret)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Genre classifier using Snorkel')
    parser.add_argument('--input', type=str, default='workshop-on-open-web-search/document-processing-20231027-training', help='Dataset id from ir_datasets/TIRA to process.')
    
    return parser.parse_args()


if __name__ == '__main__':
    dataset = ir_datasets.load(parse_args().input)

    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')
    
    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(output_dir) / 'documents.jsonl.gz'
    
    # You can pass as many additional arguments to your program, e.g., via argparse, to modify the behaviour
    
    # process the documents, store results at expected location.
    processed_documents = process_documents(dataset.docs_iter())
    processed_documents.to_json(output_file, lines=True, orient='records')
    