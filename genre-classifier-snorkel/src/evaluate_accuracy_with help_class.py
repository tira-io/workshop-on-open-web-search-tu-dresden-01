import numpy as np
from tqdm import tqdm

from load_data import load_plain_text_dfs

from genre_classification_rules_with_help_class import classifier_based_on_most_frequent_terms, classifier_based_on_most_frequent_terms_with_threshold,  label_names, classifier_based_on_vector_space

from util import preprocess_document

def run(language = "english"):
    df_train, _  = load_plain_text_dfs(language="english")

    # 1. Apply the plain text classifier to the test set
    df_train = _apply_classifier(df_train, text_col="plain_text", language=language)
    df_train_without_abstain = df_train[df_train['prediction_plain_text'] != -1]
    
    print(f'Accuracy on plain text: '
          f'{df_train.loc[df_train["label"] == df_train["prediction_plain_text"]].shape[0] / df_train.shape[0]}')

    print(f'Accuracy on plain text (without abstain): '
          f'{df_train_without_abstain.loc[df_train_without_abstain["label"] == df_train_without_abstain["prediction_plain_text"]].shape[0] / df_train_without_abstain.shape[0]}')


    # 2. Preprocess the plain text using stopword removal and lemmatization
    #tqdm.pandas(desc="Processing test data")
    #df_test["processed_text"] = df_test["plain_text"].progress_apply(lambda x: lemmatize_text(x, language))

    # 3. Apply the processed text classifier to the test data
    #df_test = _apply_classifier(df_test, text_col="processed_text", language=language)
    #print(f'Accuracy on processed text: '
    #      f'{df_test.loc[df_test["label"] == df_test["prediction_processed_text"]].shape[0] / df_test.shape[0]}')


def _apply_classifier(df, text_col: str = "plain_text", language: str = "english"):
    prediction_list = []
    for data_text in tqdm(list(df[text_col])):
        # print(data_text)
        prediction_list.append(classifier_based_on_vector_space(data_text))
        #prediction_list.append(classifier_based_on_most_frequent_terms(preprocess_document(data_text)))

    prediction_array = np.array(prediction_list)
    df[f"prediction_{text_col}"] = prediction_array
    print(df)
    return df

if __name__ == "__main__":
    run()
