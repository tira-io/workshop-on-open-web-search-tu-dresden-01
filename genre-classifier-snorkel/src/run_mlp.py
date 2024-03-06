from joblib import load
from load_data import load_plain_text_dfs
from tira.third_party_integrations import ir_datasets, get_output_directory
from __init__ import CLASSIFIER_PATH
import pandas as pd
from pathlib import Path
import enum
 
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

    v_data = vectorizer.transform(data[text_col])
    predicted_labels = mlp_classifier.predict(v_data)
    data['predicted_labels'] = predicted_labels
    return data

def get_df_text_for_ir_dataset(dataset):
    data = []
    for i in dataset.docs_iter():
        data += [{'docno': i.doc_id, 'text': i.default_text()}]
    res = pd.DataFrame(data)
    return res


if __name__ == '__main__':
    print("start the test")
    args = parse_args()

    # to classify gerneral dataset uncomment this lines and comment the lines ab 47
    dataset = ir_datasets.load(args.input)
    data = get_df_text_for_ir_dataset(dataset)
    res = classify(data)
    res = pd.DataFrame(zip(data['docno'],
                          [label(i).name for i in data['predicted_labels']]))
    output_dir = get_output_directory('.')
    output_file = Path(output_dir) / 'documents_newmlp.jsonl.gz'
    pd.DataFrame(res).to_json(output_file, lines=True, orient='records')

    # test the model on the labeled test data from zenodo dataset
    # _, dataset_test_data = load_plain_text_dfs()
    # data = dataset_test_data
    # data = classify(data, text_col='plain_text', text_type='plain_text')

    # res = pd.DataFrame(zip(data['file_name'], [label(i).name for i in data['label']], 
    #                        [label(i).name for i in data['predicted_labels']]))
    # print(f'Accuracy on test data: '
    #       f'{data.loc[data["label"] == data["predicted_labels"]].shape[0] / data.shape[0]}')
    # output_dir = get_output_directory('.')
    # output_file = Path(output_dir) / 'documents_newmlp.jsonl.gz'
    # pd.DataFrame(res).to_json(output_file, lines=True, orient='records')