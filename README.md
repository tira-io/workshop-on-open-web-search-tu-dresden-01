# Team TU-Dresden-01 at WOWS 2024: Web Page Genre Classification 

This repository contains genre classifiers submitted as retrieval component to the [Workshop on Open Web Search (WOWS) 2024](https://opensearchfoundation.org/en/events-osf/wows2024/).

## Run the software
There are two main methods implemented for genre classification:
- A rule-based genre classifier (focused on precision)
- Multilayer Perceptron Classifier

### Rule-based Genre classification:
If you want to focus on Precision, run the following:

```bash
cd PROJECT_ROOT
python3 /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/snorkel_genre_classifier.py --input path/to/dataset --rules precision
```
Result is then stored in 'documents.jsonl.gz' on TIRA_OUTPUT_DIRECTORY.

For testing the accuracy:
```bash
cd PROJECT_ROOT
python3 /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/evaluate_accuracy_snorkel.py
```

Similarly, if you want to focus on total performance, run the following:
```bash
cd PROJECT_ROOT
python3 /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/snorkel_genre_classifier.py --input path/to/dataset --rules recall
```
Result is then stored in 'documents.jsonl.gz' on TIRA_OUTPUT_DIRECTORY.
### Genre classification rules:

For testing the accuracy:
comment out line 37 and uncomment line 38 of /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/evaluate_accuracy_snorkel.py
```bash
cd PROJECT_ROOT
python3 /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/evaluate_accuracy_snorkel.py.
```
### Multilayer Perceptron Classifier


The model is already created in /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/resources/classifiers.
If you still want to train the model you can try the following:
```bash
cd PROJECT_ROOT
python 3 /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/classifier-model.py
```

The model can be used to predict new dataset as following. In /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/main.py, comment out line 26 and 35
```bash
cd PROJECT_ROOT
python3 /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/main.py --input path/to/dataset
```
Result is then stored in 'documents.jsonl.gz' on TIRA_OUTPUT_DIRECTORY

To test the accuracy, in /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/main.py, comment out line 26 and 31
```bash
cd PROJECT_ROOT
python3 /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/src/main.py --input path/to/dataset
```


## Developing in Github Codespaces

- Open this repository in Github Codespaces (i.e., click on "Code" -> "Codespaces" -> "Create ...").
- Please do not forget to commit often


## Developing in Dev Containers

A dev container (please find a suitable installation instruction [here](https://code.visualstudio.com/docs/devcontainers/containers)) allows you to directly work in the prepared Docker container so that you do not have to install the dependencies (which can sometimes be a bit tricky).

To develop with dev containers, please:

- Install [VS Code](https://code.visualstudio.com/download) and [Docker](https://docs.docker.com/engine/install/) on your machine
- Clone this repository: `git clone ...`
- Open the directory `jupyter-notebook-submissions` with VS Code (it should ask you to open the repository in a dev container)

If you do not want to use VS Code, you can start and develop in a jupyter notebook via (please execute the command within the `jupyter-notebook-submissions` directory):

```
docker run --rm  -it -p 8888:8888 --entrypoint jupyter -w /workspace -v ${PWD}:/workspace webis/ir-lab-wise-2023:0.0.1 notebook --allow-root --ip 0.0.0.0
```
