from load_data import load_plain_text_dfs
import spacy
from collections import Counter


def get_df(term, labels):
    df = 0
    for label in labels:
        if term in label:
            df += 1
    return df 

def parse_doc(doc, nlp):
    # get the doc and terms and the count of term in the doc
    doc = get_tokens_info(doc, nlp)
    doc = Counter(doc)
    new_ = {x[0]:0 for x in doc}
    for x in doc:
        new_[x[0]] += doc[x]
    return new_

def get_tokens_info(doc, nlp):
    doc = nlp(doc)
    tokens = []
    for token in doc:
        tokens.append((token.lemma_, token.is_alpha, token.is_stop))
    return [t for t in tokens if t[1] and not t[2]]

def get_tf_for_labels(df, nlp):
    labels_ids = set(df['label'])
    label_terms = {}
    for l_id in labels_ids:
        docs = df.loc[df['label'] == l_id]
        terms = {}
        my_list = []
        for i in range(len(docs)):
            doc = docs.iloc[i]
            my_list.append(parse_doc(doc["plain_text"], nlp))
        for e in my_list:
            terms = {i: e.get(i, 0) + terms.get(i, 0) for i in set(terms).union(set(e))}
        label_terms[l_id] = terms
    return label_terms

def get_term_dict(df, nlp):
    labels_ids = set(df['label'])
    term_dict = {}
    for l_id in labels_ids:
        docs = df.loc[df['label'] == l_id]
        my_list = []
        for i in range(len(docs)):
            doc = docs.iloc[i]
            dict = parse_doc(doc["plain_text"], nlp)
            my_list.append({k : [dict[k], 1] for k in dict})
        for e in my_list:
            for k in e:
                if k in term_dict:
                    term_dict[k] = [x+y for x,y in zip(term_dict[k], e[k])]
                else:
                    term_dict[k] = e[k]
    return term_dict


def run():
    df_train, _ = load_plain_text_dfs(language="english")
    nlp = spacy.load("en_core_web_sm")
    # print(get_tf_for_labels(df_train, nlp))
    # todo call get_tf_for_labels to create the tf values for each class
    labels_terms = get_tf_for_labels(df_train, nlp)
    terms = get_term_dict(df_train, nlp)
    return terms, labels_terms

    



if __name__ == "__main__":
    run()