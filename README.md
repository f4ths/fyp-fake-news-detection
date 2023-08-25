# Short-Form Fake News Detection
This project investigates the impact of various natural language processing techniques and the inclusion of metadata features on the performance of short-form news content fake news detection models, trained using the LIAR dataset. The study investigates part-of-speech (POS) tagging, dependency parsing, stop word removal, TF-IDF with K-Means, Jaccard similarity, and GloVe word embedding. Both binary and multiclass classification were investigated.

## Installation and Setup
### Python Installation and Dependencies:
This project was developed using Python version 3.9. To install Python and the required packages, there are two options:

1. Python 3.9 https://www.python.org/downloads/. Setting up PATH variable is not required. Do create a virtual environment, however. Install the dependencies using pip:
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

2. For a simpler route, download anaconda https://www.anaconda.com/download and use anaconda navigator. Create a new virtual environment with Python version 3.9. Then, install dependencies using the terminal:
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

### Dataset
The dataset used for this project was the LIAR dataset, which includes 12,836 human labelled short-form news content statements obtained from PolitiFact.com, and comes pre-split into train, valid, and test sets. 
Obtain the .tsv files of the three sets from https://paperswithcode.com/dataset/liar. 




