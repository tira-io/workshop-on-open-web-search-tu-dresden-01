from load_data import load_plain_text_dfs
from process_labels import get_tf_for_labels
import spacy
import pathlib
import json
from collections import Counter

path = pathlib.Path(__file__).parent.parent.resolve()

def save_in_file(terms, label_name):
    label_name = 'vocanulary-popsecul-modified-' + label_name + '.txt'

    with open(path / 'resources' / 'vocabulary_modified' / label_name, 'w') as out:
        for t in terms:
            out.write(t + ' ')

def create_for_label_id(l_id, nlp, df):
   
    terms = get_tf_for_labels(df.loc[df['label'] == l_id], nlp)
    filename = str(l_id) + '.json'
    with open(path / 'resources' /'Json'/ filename, "w") as outfile: 
        json.dump(terms[l_id], outfile)

    sort_terms = sorted(terms[l_id].items(), key=lambda item: -item[1])
    key_terms = [t[0] for t in sort_terms[:10]]
    save_in_file(key_terms, str(l_id))
   
def get_all_terms_df(labels_ids):
    terms = []
    for l_id in labels_ids:
        filename = str(l_id) + '.json'
        with open(path / 'resources' /'Json'/ filename, 'r') as file:
            d = json.load(file)
            terms += [k for k in d.keys()]
    
    return Counter(terms)

if __name__ == '__main__':

    _, df_test = load_plain_text_dfs(language="english")

    nlp = spacy.load("en_core_web_sm")
    
    labels_id = df_test['label']
    for label_id in labels_id:
        create_for_label_id(label_id, nlp, df_test)
    all = get_all_terms_df(labels_id)
    with open(path / 'resources' /'Json'/ 'all_terms.json', "w") as outfile: 
        json.dump(all, outfile)
    
    