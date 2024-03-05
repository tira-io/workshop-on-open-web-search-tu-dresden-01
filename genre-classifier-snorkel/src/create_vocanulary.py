from load_data import load_plain_text_dfs
from process_labels import get_tf_for_labels
import spacy
import pathlib

path = pathlib.Path(__file__).parent.parent.resolve()

def save_in_file(terms, label_name):
    label_name = 'vocanulary-popsecul-modified-' + label_name + '.txt'

    with open(path / 'resources' / 'vocabulary' / label_name, 'w') as out:
        for t in terms:
            out.write(t + ' ')

def create_for_label_id(l_id, nlp, df):
   
    terms = get_tf_for_labels(df.loc[df['label'] == l_id], nlp)
    sort_terms = sorted(terms[3].items(), key=lambda item: -item[1])
    key_terms = [t[0] for t in sort_terms[:10]]
    
    save_in_file(key_terms, 'help')
    return

if __name__ == '__main__':
    _, df_test = load_plain_text_dfs(language="english")

    nlp = spacy.load("en_core_web_sm")
    create_for_label_id(3, nlp, df_test)