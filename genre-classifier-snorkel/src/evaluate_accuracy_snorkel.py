import numpy as np
from tqdm import tqdm
from load_data import load_plain_text_dfs
from genre_classification_rules import classifier_based_on_most_frequent_terms, classifier_based_on_most_frequent_terms_with_threshold,  ABSTAIN, DISCUSSION, SHOP, DOWNLOAD, ARTICLES, HELP, LINKLISTS, PORTRAIT_PRIV, PROTAIT_NPRIV
from util import preprocess_document
from sklearn.metrics import classification_report

def run(language = "english"):
    _, df_test = load_plain_text_dfs(language=language)
    df_test = df_test[df_test["label"] != HELP]

    # 1. Apply the plain text classifier to the test set
    df_test = _apply_classifier(df_test, text_col="plain_text", language=language)
 
    print(classification_report(df_test['label'], df_test['prediction_plain_text']))

def _apply_classifier(df_test, text_col: str = "plain_text", language: str = "english"):
    prediction_list = []
    for data_text in tqdm(list(df_test[text_col])):
        # choos the method you want to use
        # prediction_list.append(classifier_based_on_most_frequent_terms_with_threshold(preprocess_document(data_text)))
        prediction_list.append(classifier_based_on_most_frequent_terms(preprocess_document(data_text)))

    prediction_array = np.array(prediction_list)
    df_test[f"prediction_{text_col}"] = prediction_array
    return df_test

if __name__ == "__main__":
    run()





