import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import pyprind
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
# from tensorflow import keras
# import tensorflow as tf

data_path = 'training_data/smaller_tweets.csv'
stop_words = stopwords.words('english')

# preprocessor for text filters out unhelpful words and puts all in lowercase
def preprocessor(text):
    text = text.lower()
    # remove words that contain numbers
    text = re.sub(r'(\b\d+\w*)|(\w*\d+\b)|(\w*\d+\w*)', '', text)
    # remove links
    text = re.sub(
        r'(http|https)?(:\/\/)?\w+(\.)?\w+\.(com|co|ly|net|org).+\b', '', text)
    # replace punctuation with spaces
    text = re.sub(r'[\W]', ' ', text)
    return text

# simple tokenizer with stop word deletion
def tokenizer_stopper(text):
    global stop_words
    return [w for w in text.split() if w not in stop_words]


def stream_docs(path, csv_encoding='ISO-8859-1'):
    with open(path, 'r', encoding=csv_encoding) as csv:
        next(csv)
        for line in csv:
            line = re.sub(r'^\d+\,', '', line)
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

def train_classifier(data_path, num_batches=100, test_frac=0.2):
    # prepare for online learning
    sample_count = len(open(data_path).readlines())
    train_samples = round(sample_count * (1 - test_frac))
    test_size = sample_count - train_samples - 1
    train_size = round(train_samples / num_batches)
    pbar = pyprind.ProgBar(num_batches)
    classes = np.array([0, 4])

    # initialize vectorizer and classifier
    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2**18,
                             preprocessor=preprocessor,
                             tokenizer=tokenizer_stopper)


    clf = SGDClassifier(loss='log', max_iter=1000)
    doc_stream = stream_docs(path=data_path)

    # perform training
    for _ in range(num_batches - 5):
        X_train, y_train = get_minibatch(doc_stream, size=train_size)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()

    X_test, y_test = get_minibatch(doc_stream, size=test_size)
    X_test = vect.transform(X_test)
    print('\nAccuracy: %.3f' % clf.score(X_test, y_test))

    return vect, clf
