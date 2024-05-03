# TPDL-2024
## Introduction
RetractionCascade is a project aimed at analyzing and understanding academic paper retractions and their cascading effects. This project utilizes various natural language processing (NLP) and machine learning techniques to parse XML files, calculate similarities between texts, classify papers using BERT and LSTM models, and predict retraction using decision trees.

## Contents
XMLParsing.py
Similarity.py
BertClassification.py
LSTMClassification.py
DT.py

## Dataset
It includes details of articles citing a retracted article, including PubMed Central IDs, title, authors' list, in-text article information, citation sentence, frequency of reference to the retracted article, citation intention, list of cited references, self-citation flag, matched references between bibliographically coupled articles, and retraction flag.

## XMLParsing.py
This script parses XML files containing academic paper data, extracts relevant information such as titles, authors, abstracts, sections, and references, and stores the data in a structured format in a CSV file.

## Similarity.py
Similarity.py calculates similarities between academic papers based on various features such as author names, cited references, abstracts, and citation sentences. It utilizes Sentence-BERT (SBERT) for embedding text and computes Jaccard similarity and cosine similarity metrics.

## BertClassification.py
BertClassification.py performs binary classification of academic papers using the BERT model. It preprocesses text data, encodes it using BERT tokenizer, trains the BERT-based classifier, and evaluates its performance using metrics such as accuracy and classification report.

## LSTMClassification.py
LSTMClassification.py implements a Long Short-Term Memory (LSTM) neural network for binary classification of academic papers. It preprocesses text data, vectorizes it using CountVectorizer, trains the LSTM classifier, and evaluates its performance using metrics such as accuracy and classification report.

## DT.py
DT.py utilizes a Decision Tree classifier to predict the retraction status of academic papers based on features such as self-citations, pre-retraction citations, similar references, and citation frequency. It visualizes the decision tree and provides a classification report.

## Installation
To run the scripts in this project, you need Python installed on your system along with the necessary libraries listed in the requirements.txt file. You can install the dependencies using pip:

### bash
pip install -r requirements.txt



## Usage
Each script can be run independently by executing the corresponding Python file. Make sure to provide the required input data files as specified within the scripts.

## Acknowledgements
The code snippets in this project utilize various open-source libraries and tools.
Special thanks to the authors and contributors of these libraries.


