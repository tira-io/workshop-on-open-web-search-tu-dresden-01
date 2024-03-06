#!/usr/bin/env python3
# Load a patched ir_datasets that loads the injected data inside the TIRA sandbox
from tira.third_party_integrations import ir_datasets, get_output_directory
from pathlib import Path
import pandas as pd
from snorkel.labeling import LabelingFunction, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from tqdm import tqdm
from genre_classification_rules import classifier_based_on_most_frequent_terms, classifier_based_on_most_frequent_terms_with_threshold, label_names
from util import preprocess_document

def get_snorkel_pandas_lf_applier(variant=None):
    lfs = [classifier_based_on_most_frequent_terms, classifier_based_on_most_frequent_terms_with_threshold,
          ]
    if variant == 'precision':
        print('Use precision rules')
        lfs = [classifier_based_on_most_frequent_terms_with_threshold]
    if variant == 'recall':
        print('Use recall rules')
        lfs = [classifier_based_on_most_frequent_terms]

    return PandasLFApplier(lfs=[LabelingFunction(name=func.__name__, f=func) for func in lfs])

def process_documents(document_iter, variant=None):
    snorkel_applier = get_snorkel_pandas_lf_applier(variant)
    df = []
    for i in tqdm(document_iter, 'Pre-process Documents.'):
        doc = preprocess_document(i.default_text())
        doc['docno'] = i.doc_id
        doc['text'] = i.default_text()
        df += [doc]

    df = pd.DataFrame(df)
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
    parser.add_argument('--rules', type=str, default=None, help='Set to none or precision or recall.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = ir_datasets.load(args.input)

    # The expected output directory, injected via the environment variable TIRA_OUTPUT_DIRECTORY
    output_dir = get_output_directory('.')
    
    # Document processors persist their results in a file documents.jsonl.gz in the output directory.
    output_file = Path(output_dir) / 'documents.jsonl.gz'
    
    # You can pass as many additional arguments to your program, e.g., via argparse, to modify the behaviour
    
    # process the documents, store results at expected location.
    processed_documents = process_documents(dataset.docs_iter(), args.rules)
    processed_documents.to_json(output_file, lines=True, orient='records')
    