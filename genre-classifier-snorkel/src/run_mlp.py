from joblib import load
from load_data import load_plain_text_dfs
from tira.third_party_integrations import ir_datasets, get_output_directory
from __init__ import CLASSIFIER_PATH
from utils import lemmatize_text
import pandas as pd
from pathlib import Path
import enum
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# in this file is the funktion to use the mlp classifier 
 
class label(enum.Enum):
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

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Genre classifier using MLP')
    parser.add_argument('--input', type=str, default='workshop-on-open-web-search/document-processing-20231027-training', help='Dataset id from ir_datasets/TIRA to process.')
    return parser.parse_args()

def classify(data, modelname='mlp', language='english', text_col='text', text_type='plain_text'):
    
    vectorizer = load(CLASSIFIER_PATH / f"vectorizer_{text_type}_{modelname}_{language}.joblib")
    mlp_classifier = load(CLASSIFIER_PATH / f"classifier_{text_type}_{modelname}_{language}.joblib")

    if text_type == 'processed_text':
        data["processed_text"]  = data["plain_text"].progress_apply(lambda x: lemmatize_text(x, language))
    v_data = vectorizer.transform(data[text_col])
    predicted_labels = mlp_classifier.predict(v_data)
    prob_prediction = mlp_classifier.predict_proba(v_data)
    data['predicted_labels'] = predicted_labels
    data['prob_prediction'] = [max(i) for i in prob_prediction]
    
    # if want to remove predicted labels with prob < 0.5
    # data = data[data['prob_prediction'] >= 0.5]
    return data

def get_df_text_for_ir_dataset(dataset):
    data = []
    for i in dataset.docs_iter():
        data += [{'docno': i.doc_id, 'text': i.default_text()}]
    res = pd.DataFrame(data)
    return res

def run_mlp(dataset, text_type='plain_text'):
    data = get_df_text_for_ir_dataset(dataset)
    res = classify(data, text_type = text_type)
    res = pd.DataFrame(zip(data['docno'],
                           [label(i).name for i in data['predicted_labels']]), columns=['docno', 'label'])
    return res

def run_mlp_test_data():
    _, dataset_test_data = load_plain_text_dfs()
    data = dataset_test_data
    data = classify(data, text_col='plain_text', text_type='plain_text')

    res = pd.DataFrame({ 'file_name': data['file_name'], 'label': [label(i).name for i in data['label']], 
                          'predicted_labels':[label(i).name for i in data['predicted_labels']],
                          'prob_prediction': data['prob_prediction']})
    
    print(classification_report(res['label'], res['predicted_labels']))
    
    return res


if __name__ == '__main__':
    print("Run MLP classifier.")
    args = parse_args()

    # run mlp on input dataset
    # dataset = ir_datasets.load(args.input)
    # res = run_mlp(dataset)

    # test the model on the labeled test data from zenodo dataset
    res = run_mlp_test_data()
    
    # save the results in file 
    output_dir = get_output_directory('.')
    output_file = Path(output_dir) / 'documents_newmlp.jsonl.gz'
    pd.DataFrame(res).to_json(output_file, lines=True, orient='records')
