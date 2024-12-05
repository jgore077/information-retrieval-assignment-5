# Information Retrieval Assignment 5
This is my code for assignment 5, I decided to use a bi-encoder because they are the best models we have seen in the course. I specifically used the `all-mpnet-base-v2` bi-encoder because it's at the top of the [leadboard](https://sbert.net/docs/sentence_transformer/pretrained_models.html#original-models).

# Installation
Install the python packages.
```
pip install -r requirements.txt
```
# Scripts
This repo has a few scripts you can run, the other python files are just classes.

## `results.py`
This file is used to compute the results for the bi-encoders on a topics file.
It can either be used to compute the results of every bi-encoder in `models.txt`
or you can provide a name & path to compute the results for one.

```
python results.py data/Answers.json <topics_path>.json
```
```
python results.py data/Answers.json <topics_path>.json <model_name> <output_path>.tsv
```

## `compare.py`
This file is used to compare the results files together. You need to be sure that the directory only contains results files that all use the same qrel.
```
python compare.py <qrel_path>.tsv <results_directory>
```
It will then print the P@10 sorted in descending order for every result file in th results directory.

## `rerank.py`

This script was used to re-rank bi-encoder results with a cross-encoder. The results are actually worse so this was a failed experiment.
```
------------------------------------------
Model Name                        |   P@10
------------------------------------------
all-mpnet-base-v2                 | 0.4357
------------------------------------------
all-mpnet-base-v2-re-ranked       | 0.2152
-------------------------------------------
```
However if you want to use this script this is the usage.
```
python rerank.py <results_path>.tsv <topics_path>.json <output_path>.tsv
```