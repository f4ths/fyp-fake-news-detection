# Short-Form Fake News Detection
This project investigates the impact of various natural language processing techniques and the inclusion of metadata features on the performance of short-form news content fake news detection models, trained using the LIAR dataset. The study investigates part-of-speech (POS) tagging, dependency parsing, stop word removal, TF-IDF with K-Means, Jaccard similarity, and GloVe word embedding. Both binary and multiclass classification were investigated.

## Installation and Setup
### Python Installation and Dependencies:
This project was developed using Python version 3.9. To install Python and the required packages, there are two options:

1. Install Python 3.9 from https://www.python.org/downloads/. Setting up PATH variable is not required. Please do create a virtual environment, however.  

Install the dependencies using pip:
```
pip install scipy==1.10.0
pip install scikit-learn==1.2.1
pip install tensorflow==2.10.0
pip install spacy==3.3.1
pip install matplotlib==3.6.2
pip install keras==2.10.0 nltk==3.7
pip install tensorboard==2.10.0
pip install pandas==1.5.2
pip install numpy==1.23.5
pip install pydot==1.4.2
pip install jupyterlab==3.5.3
pip install cupy==8.3.0
pip install seaborn==0.12.2
pip install jupyterlab-git==0.41.0
pip install wordcloud==1.8.2.2
```

2. For a simpler route, use anaconda navigator which can be installed from https://www.anaconda.com/download. Create a new virtual environment with Python version 3.9.  

Install dependencies using the terminal:
```
conda install -c anaconda scipy=1.10.0 
conda install -c anaconda scikit-learn=1.2.1 
conda install -c anaconda tensorflow=2.10.0 
conda install -c anaconda spacy=3.3.1 
conda install -c anaconda matplotlib=3.6.2 
conda install -c anaconda keras=2.10.0 
conda install -c anaconda nltk=3.7 
conda install -c anaconda tensorboard=2.10.0 
conda install -c anaconda pandas=1.5.2 
conda install -c anaconda numpy=1.23.5 
conda install -c anaconda pydot=1.4.2 
conda install -c anaconda jupyterlab=3.5.3 
conda install -c anaconda cupy=8.3.0 
conda install -c anaconda seaborn=0.12.2 
conda install -c anaconda jupyterlab-git=0.41.0 
conda install -c anaconda wordcloud=1.8.2.2
```

### Dataset
The dataset used for this project was the LIAR dataset, which includes 12,836 human labelled short-form news content statements obtained from PolitiFact.com, and comes pre-split into train, valid, and test sets as train.tsv, valid.tsv, and test.tsv.


Obtain the .tsv files of the three sets from https://paperswithcode.com/dataset/liar.

### GloVe word embeddings
GloVe was used as a word embedding method for the models. To obtain the necessary files, visit https://nlp.stanford.edu/projects/glove/ and download "Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip "


Only glove.6B.300d.txt and glove.6B.100d.txt were used.

## Working files
### preprocessing.ipynb
This notebook prepares the LIAR dataset for model implementation:

Dataset Preparation:
- Converted train.tsv, valid.tsv, and test.tsv to pandas dataframes.
- Handled missing values in specific columns and dropped the 'state' column.
- Mapped news statement labels to a scale from 0 (pants-fire) to 5 (true).
- Mapped statements to binary classification: 0 for false news and 1 for true news.

    Text Data Preprocessing:
        Applied spaCy for lemmatization, lowercasing, non-alphanumeric character removal, and stop word elimination.
        Tokenized statements and generated two vocabulary dictionaries: one for spaCy's predefined stop words and another for custom stop words based on POS tags.
        Converted tokenized statements into numerical representations for BiLSTM-GRU model input.
        Utilized 300d pretrained GloVe embeddings to generate embedding matrices for the BiLSTM-GRU model.

    POS and DEP Processing:
        Extracted POS and DEP tags using spaCy's NLP model, adding them as features in dataframes.
        Generated three sets of POS tags: fine-grained, coarse-grained, and a reduced set.
        Two DEP dictionaries were derived: one based on a reference implementation and another with additional tags to enhance Fake News Detection (FND).

    Metadata Clustering:
        Implemented two clustering methods: TF-IDF K-means (focused on the 'party' column) and Jaccard similarity clustering with specific cluster distribution for each metadata column.
        Converted processed metadata into numerical values for the Dense model input.

### bilstm.ipynb 
First deep learning model that was worked on. However, because there was no literature support for the model, it was later scrapped for ensemble.ipynb.




