# HomeWork 2 -- GTZAN Music Genre Classification

This repo provides an implementation of CS5491 Programming Homework 2 (City University of Hong Kong) -- TZAN Music Genre Classification

## Datasets
The datasets are provided by [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

## Experimental Dependency (python)
```shell
# create virtual environment
conda create --name=CS5491 python=3.9

# activate virtual environment
conda activate CS5491

# install dependencies
pip install -r requirements.txt
```

## Usage
```shell
# Step 1: Generate data
python data_extractor.py

# Step 2: Baselines
python baseline.py

# Step 3: CNN
python main.py --aug 0

# Step 4: CNN + Mixup
python main.py --aug 1

# Step 5: Confusion Matrix
python test.py
```
