from joblib import load
from tira.third_party_integrations import ir_datasets, get_output_directory
from __init__ import CLASSIFIER_PATH
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
 

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Genre classifier using MLP')
    parser.add_argument('--input', type=str, default='workshop-on-open-web-search/document-processing-20231027-training', help='Dataset id from ir_datasets/TIRA to process.')
    return parser.parse_args()

def classify(data, modelname='mlp', language='english', text_col='plain_text'):
    
    vectorizer = load(CLASSIFIER_PATH / f"vectorizer_{text_col}_{modelname}_{language}.joblib")
    mlp_classifier = load(CLASSIFIER_PATH / f"classifier_{text_col}_{modelname}_{language}.joblib")

    v_data = vectorizer.transform(data['text'])
    predicted_labels = mlp_classifier.predict(v_data)
    data['labels'] = predicted_labels
    return data

def get_df_text(dataset):
    doc = {}
    data = []
    for i in dataset.docs_iter():
        data += [{'docno': i.doc_id, 'text': i.default_text()}]
    res = pd.DataFrame(data)
    return res

if __name__ == '__main__':
    print("start the test")
    args = parse_args()
    dataset = ir_datasets.load(args.input)
    data = get_df_text(dataset)
    print(data)
    res = classify(data)
    output_dir = get_output_directory('.')
    output_file = Path(output_dir) / 'documents_newmlp.jsonl.gz'
    pd.DataFrame(data).to_json(output_file, lines=True, orient='records')