#!/bin/bash

pwd=$(dirname $(readlink -f $0))

if test -d "${pwd}/english_corpus"; then
  echo "The corpus already exists under ${pwd}/english_corpus. Delete this directory if you want to create it anew."
  exit 0
fi

# Download the file
wget -nc -P "${pwd}" https://zenodo.org/records/3233881/files/corpus-genre-ki-04.zip

# Unzip the corpus
sudo apt-get install unzip
unzip "${pwd}/corpus-genre-ki-04.zip" -d "${pwd}/"
mv "${pwd}/genre-corpus-04/" "${pwd}/english_corpus/"