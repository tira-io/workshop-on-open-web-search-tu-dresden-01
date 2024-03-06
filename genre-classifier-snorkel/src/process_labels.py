from collections import Counter

# in this file is the preprocessing for the documents lemmatizing and counting the frequency

def parse_doc(doc, nlp):
    # get the doc and terms and the count of term in the doc
    doc = get_tokens_info(doc, nlp)
    doc = Counter(doc)
    new_ = {x[0]:0 for x in doc}
    for x in doc:
        new_[x[0]] += doc[x]
    return new_

def get_tokens_info(doc, nlp):
    if type(doc) != str:
        doc = doc.text
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
