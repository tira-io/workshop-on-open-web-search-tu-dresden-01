from pathlib import Path
import codecs
import pandas as pd
from resiliparse.extract.html2text import extract_plain_text
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.resolve() / "data"

def load_plain_text_dfs(language="english"):
    data_path = DATA_PATH / f"{language}_corpus"
    train_tuples = _load_file_tuples(data_path=data_path.parent, file_stem="train_split.txt")
    test_tuples = _load_file_tuples(data_path=data_path.parent, file_stem="test_split.txt")

    df_train = pd.DataFrame([_load_plain_text_file(data_path=data_path, file_tup=file_tup) for file_tup in train_tuples])
    df_test = pd.DataFrame([_load_plain_text_file(data_path=data_path, file_tup=file_tup) for file_tup in test_tuples])

    return df_train, df_test


def _load_file_tuples(data_path, file_stem="train_split.txt"):
    with open(data_path / file_stem, "r") as f:
        lines = f.readlines()
    num_genres = int(lines[0])

    file_tuples = [x.split(";") for x in lines[num_genres + 1:]]
    file_tuples = [(int(x[0]), x[1].replace("\n", "")) for x in file_tuples]

    return file_tuples

def _load_plain_text_file(data_path, file_tup):
    label, file_stem = file_tup

    html = codecs.open(data_path / file_stem, 'r').read()
    plain_text = extract_plain_text(html)
    return {"label": label, "file_name": file_stem, "plain_text": plain_text}