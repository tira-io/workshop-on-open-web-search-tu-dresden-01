# Genre Classifier Baseline
This code is the basis to develop a genre classifier for web pages. 
It provides a [download script](data/prepare_dataset.sh) for the dataset and two sample classifiers to be used as baselines.

## Run the Genre Classifier

In the end, the classifier has to work in the following script:

```
python3 genre-classifier-snorkel/src/snorkel_genre_classifier.py
```

This produces an output like (output via zcat documents.jsonl.gz`):

```
{"docno":"doc-1","label":"Abstain"}
{"docno":"doc-2","label":"Abstain"}
```

This is intended to run on arbitrary content from ir_datasets, e.g., you can pass an arbitrary dataset id from ir_datasets.com for --input. e.g., for the cranfield collection:

```
python3 genre-classifier-snorkel/src/snorkel_genre_classifier.py --input cranfield
```



## Preparation
### Dataset
The dataset can be downloaded from [Zenodo](https://zenodo.org/records/3233881). 
You can perform the following steps manually or use the bash script:

**Script command**
```
./data/prepare_dataset.sh
```

**Manual steps**
1. Download the zip-File from Zenodo
2. Place the file in the [data](data) directory
3. Unzip the file and rename the directory "genre-corpus-04" to "english_corpus"

### Dependencies
All dependencies for the baseline classifier are found in [requirements.txt](requirements.txt).

## Resources
The [resources](resources) directory provides some additional files for developing a classifier:

- [classifiers](resources/classifiers): Directory to save and load classifier weights
- [vocabulary](resources/vocabulary): Manually gathered collection of stemmed terms associated with each of the 8 genres.
- [wordlists](resources/wordlists): Text files of terms associated with a specific area (e.g. `countries.txt`) 

## Sample classifier
The process of developing a genre classifier is illustrated using a Naive Bayes classifier. 
The training code can be found in [train.py](src/train.py), it trains two classifiers:

1. Naive Bayes on the vectorized plain text of the web pages (`plain_text`)
2. Naive Bayes on the vectorized text of the web pages after removing stop words and applying lemmatization (`processed_text`)

### Training
The trained classifiers are shared with this repository under (resources/classifiers). If you want to do the training by yourself, run
```
./src/train.py
```

### Testing
To get the accuracy on the test data, run
```
./src/test.py
```

### Results
Adding lemmatization improves the accuracy a little, suggesting that sklearn`s vectorization works already well on English web pages.

| Classifier     | Accuracy |
|----------------|----------|
| Plain text     | 0.6174   |
| Processed text | 0.6495   |
