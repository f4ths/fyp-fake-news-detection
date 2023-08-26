# Short-Form Fake News Detection
This project investigates the impact of various natural language processing techniques and the inclusion of metadata features on the performance of short-form news content fake news detection models, trained using the LIAR dataset. The study investigates part-of-speech (POS) tagging, dependency parsing, stop word removal, TF-IDF with K-Means, Jaccard similarity, and GloVe word embedding. Both binary and multiclass classification were investigated.

# Installation and Setup
## Python Installation and Dependencies:
This project was developed using Python version 3.9. To install Python and the required packages, there are two options:

1. Install Python 3.9 from https://www.python.org/downloads/. Setting up PATH variable is not required. Virtual environment is recommended.

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

2. For a simpler method, use anaconda navigator which can be installed from https://www.anaconda.com/download. Create a new virtual environment with Python version 3.9.  

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

## Dataset
The dataset used for this project was the LIAR dataset, which includes 12,836 human labelled short-form news content statements obtained from PolitiFact.com, and comes pre-split into train, valid, and test sets as train.tsv, valid.tsv, and test.tsv.


Obtain the .tsv files of the three sets from https://paperswithcode.com/dataset/liar.

## GloVe word embeddings
GloVe was used as a word embedding method for the models. To obtain the necessary files, visit https://nlp.stanford.edu/projects/glove/ and download "Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip "


Only glove.6B.300d.txt and glove.6B.100d.txt were used.

# Working Files
### preprocessing.ipynb
This notebook prepares the LIAR dataset for model implementation:

Dataset Preparation:
- Converted train.tsv, valid.tsv, and test.tsv to pandas dataframes.
- Handled missing values in specific columns and dropped the 'state' column.
- Mapped news statement labels to a scale from 0 (pants-fire) to 5 (true).
- Mapped statements to binary classification: 0 for false news and 1 for true news.

Text Data Preprocessing:
- Applied spaCy for lemmatization, lowercasing, non-alphanumeric character removal, and stop word elimination.
- Tokenized statements and generated two vocabulary dictionaries: one for spaCy's predefined stop words and another for custom stop words based on POS tags.
- Converted tokenized statements into numerical representations for BiLSTM-GRU model input.
- Utilized 300d pretrained GloVe embeddings to generate embedding matrices for the BiLSTM-GRU model.

POS and DEP Processing:
- Extracted POS and DEP tags using spaCy's NLP model, adding them as features in dataframes.
- Generated three sets of POS tags: fine-grained, coarse-grained, and a reduced set.
- Two DEP dictionaries were derived: one based on a reference implementation and another with additional tags to enhance Fake News Detection (FND).

Metadata Clustering:
- Implemented two clustering methods: TF-IDF K-means (focused on the 'party' column) and Jaccard similarity clustering with specific cluster distribution for each metadata column.
- Converted processed metadata into numerical values for the Dense model input.

## ensemble.ipynb
This notebook delves into the deep learning ensemble model training and testing. Derived from the foundation established by Aslam et al., the notebook explores variations and improvements upon this baseline. 

Key points:
- The ensemble model derives its core architecture from Aslam et al. However, due to the unique preprocessing techniques explored in this study, certain adjustments were essential.
- All models have certain generic variations, discussed in the sections below.
- This notebook delves into specific changes for the Dense model, taking metadata, POS, and DEP tags as input, and the BiLSTM-GRU model, particularly focusing on different stop word removal methods for text data.

### Model Preprocessing & Configuration
1. Input Sequence Length: The pad length for all sequences was standardized to 35, derived from the average sequence length of statements. Notably, some outliers existed, such as the statement at row 1280, with a sequence length of 786.
2. Callbacks: Added 'EarlyStopping' as an additional callback to mitigate overfitting. This was particularly useful, as the models frequently overfit after the epoch numbers specified by Aslam et al. The focus was to monitor validation loss instead of validation accuracy, based on empirical observations and Keras recommendations.
3. Learning Rate: While Aslam et al. didn't specify a learning rate, our observations led to choosing 0.0005 for Dense models and 0.00025 for BiLSTM-GRU models.
4. Hyperparameters: Most were kept consistent with Aslam et al., including epoch, batch size, activation functions, and optimization algorithms.
5. Model Variations: Introduced multiclass models, L2 regularization for Dense models to counter overfitting, and dropout layers in both BiLSTM and GRU layers for the BiLSTM-GRU model. Class weights provided by Aslam et al. were also tested.

### Fully Connected Dense Model
![densestrucutrer](https://github.com/f4ths/fyp-fake-news-detection/assets/91867823/500b8719-d19a-434e-bf05-601e348dc0f8)


Features:
- Inclusion of POS and DEP tags in addition to metadata input.
- Implementation of L2 kernel regularization function (0.001) to every Dense layer.
- Patience was set to 20, found ideal for displaying overfitting trends while managing training time.
- Multiclass models had an output shape of 6, whereas binary models had an output shape of 2.
- Best inputs: For multiclass, pos-id-fine provided top accuracy; for binary, pos-id-spacy was superior.

### BiLSTM-GRU Model
![bilstmgrustrcuture](https://github.com/f4ths/fyp-fake-news-detection/assets/91867823/97c5a26e-271d-4557-9003-a59fa2f45ade)


Features:
- The model utilizes dropout layers (0.3) in both BiLSTM and GRU layers.
- word-id-spacy and word-id-custom were tested as model inputs.
- word-id-spacy was preferable for binary models due to better F1 scores and improvements in end-of-training performance.
- Class weights were tested but were later discarded due to marginal performance improvements.

### Ensemble Model with Majority Voting
Methodology:
- The ensemble approach combined predictions from binary classification Dense and BiLSTM-GRU models using majority voting.
- Predictions were binarized using a 0.5 threshold.
- The Dense model utilized pos-id-spacy, dep-id-custom, and Jaccard metadata as inputs, while the BiLSTM-GRU model employed word-id-spacy.

## bilstm.ipynb and eda.ipynb
First deep learning model that was worked on. However, because there was no literature support for the model, it was later scrapped.




