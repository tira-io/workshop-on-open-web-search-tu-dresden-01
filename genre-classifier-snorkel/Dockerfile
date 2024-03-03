#see ../Dockerfile.dev for the devcontainer
FROM webis/wows-2024-baseline:0.0.1-snorkel

ADD data /app/data
ADD resources /app/resources
ADD src /app/src
ADD requirements.txt /app
ADD tests /app/tests
ADD tests /workspaces/workshop-on-open-web-search-tu-dresden-01/genre-classifier-snorkel/tests

RUN pip3 install -r /app/requirements.txt \
    && cd /app \
    && pytest

ENTRYPOINT [ "python3", "/app/src/snorkel_genre_classifier.py", "--input", "$inputDataset" ]
