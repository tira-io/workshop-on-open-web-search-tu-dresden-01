from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from load_data import load_plain_text_dfs
from joblib import dump
from tqdm import tqdm
from utils import lemmatize_text
from __init__ import CLASSIFIER_PATH

def run(language = "english"):
    df_train, _ = load_plain_text_dfs(language=language)

    # 1. Train a Naive Bayes Classifier directly on the plain text
    _train_classifier(df_train, text_col="plain_text", language=language)

    # 2. Preprocess the plain text using stopword removal and lemmatization
    tqdm.pandas(desc="Processing training data")
    df_train["processed_text"] = df_train["plain_text"].progress_apply(lambda x: lemmatize_text(x, language))

    # 3. Train a Naive Bayes Classifier on the processed text
    _train_classifier(df_train, text_col="processed_text", language=language)


def _train_classifier(df_train, text_col: str = "plain_text", language: str = "english"):
    # Vectorize the text
    vectorizier = TfidfVectorizer()
    X_train = vectorizier.fit_transform(df_train[text_col])
    y_train = df_train['label']
    dump(vectorizier, CLASSIFIER_PATH / f"vectorizer_{text_col}_{'mlp'}_{language}.joblib")

    # Train and save the classifier
    nb_classifier = MLPClassifier(hidden_layer_sizes=(50))
    nb_classifier.fit(X_train, y_train)

    out_path = CLASSIFIER_PATH / f"classifier_{text_col}_{'mlp'}_{language}.joblib"
    dump(nb_classifier, out_path)
    print(f"Trained classifier on plain text and saved results to {out_path}\n")


if __name__ == "__main__":
    run()