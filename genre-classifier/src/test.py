import numpy as np
from joblib import load
from tqdm import tqdm

from load_data import load_plain_text_dfs
from utils import lemmatize_text
from src import CLASSIFIER_PATH

def run(language = "english"):
    _, df_test = load_plain_text_dfs(language=language)

    # 1. Apply the plain text classifier to the test set
    df_test = _apply_classifier(df_test, text_col="plain_text", language=language)
    print(f'Accuracy on plain text: '
          f'{df_test.loc[df_test["label"] == df_test["prediction_plain_text"]].shape[0] / df_test.shape[0]}')

    # 2. Preprocess the plain text using stopword removal and lemmatization
    tqdm.pandas(desc="Processing test data")
    df_test["processed_text"] = df_test["plain_text"].progress_apply(lambda x: lemmatize_text(x, language))

    # 3. Apply the processed text classifier to the test data
    df_test = _apply_classifier(df_test, text_col="processed_text", language=language)
    print(f'Accuracy on processed text: '
          f'{df_test.loc[df_test["label"] == df_test["prediction_processed_text"]].shape[0] / df_test.shape[0]}')


def _apply_classifier(df_test, text_col: str = "plain_text", language: str = "english"):
    vectorizer = load(CLASSIFIER_PATH / f"vectorizer_{text_col}_{language}.joblib")
    nb_classifier = load(CLASSIFIER_PATH / f"nb_clf_{text_col}_{language}.joblib")

    X_test = vectorizer.transform(df_test[text_col])
    probs = nb_classifier.predict_proba(X_test)
    df_test[f"prediction_{text_col}"] = np.argmax(probs, axis=1)
    return df_test

if __name__ == "__main__":
    run()





