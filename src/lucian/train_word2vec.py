from used_repos.personal.aggregated_personal_repos.Cross_domain_NER.src.common.util import get_data, get_all_data
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from gensim.test.utils import common_texts
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import multiprocessing
import gensim
import random
import nltk
import pdb


stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def document_preprocess(document):
    """

    :param document:
    :return:
    """
    return word_tokenize(document)


def load_liro_dataset():
    """

    :return:
    """
    docs = np.load(file="data/all_documents_list.npy", allow_pickle=True)
    return docs


def prepare_data():
    """

    :return:
    """
    data, _ = get_all_data(first_n=100)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]
    docs = np.array([datapoint["tokens"] for datapoint in (train + valid + test)])
    np.save(file="data/all_documents_list.npy",
            arr=docs,
            allow_pickle=True)
    print("saved docs")


def train_word2vec(docs):
    """

    :param docs:
    :return:
    """
    model = Word2Vec(sentences=docs, vector_size=150, window=5, min_count=1,
                     workers=multiprocessing.cpu_count())  # an word2vec model is trained
    model.save("checkpoints/word2vec.model")  # checkpoint is saved
    print("saved model")


def load_word2vec():
    """

    :return:
    """
    return gensim.models.Word2Vec.load("checkpoints/word2vec.model")


def ensemble_voting(X_train, y_train, X_test, y_test):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    estimators = [
        ("svc", SVC(class_weight="balanced")),
        ("random_forest", RandomForestClassifier(class_weight="balanced")),
        ("decision_tree", DecisionTreeClassifier(class_weight="balanced")),
        ("xgb", XGBClassifier())
    ]
    ensemble = Pipeline(steps=[("voter", VotingClassifier(estimators))])
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    print("Ensemble clf score:", f1_score(y_pred, y_test, average="weighted"))


def train_classifier_head(X_train, y_train, X_test, y_test):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    clfs = [
        ("logistic", SVC(class_weight="balanced")),
        ("random_forest", RandomForestClassifier(class_weight="balanced")),
        ("decision_tree", DecisionTreeClassifier(class_weight="balanced")),
        ("xgb", XGBClassifier())
    ]
    for (name, clf) in clfs:
        print(X_train.shape)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        y_pred = np.zeros(shape=y_test.shape)
        test_score = f1_score(y_test, y_pred, average="weighted")
        print("*" * 10)
        print(f"F1 score: {test_score} - CLF: {name}")
        print("*" * 10)


def embed(text, word2vec_model):
    """

    :param text:
    :param word2vec_model:
    :return:
    """
    try:
        vector = word2vec_model.wv[document_preprocess(text)]
        vector = np.mean(vector, axis=0)
        vector = np.reshape(vector, (1, vector.shape[0]))
    except KeyError:
        vector = np.random.rand(1, 150)
    return vector


def create_train_test_data(word2vec_model):
    """

    :param word2vec_model:
    :return:
    """
    data, _ = get_all_data(first_n=100)
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

    train_valid = train + valid

    vectorized_train = np.array([datapoint["tokens"] for datapoint in train_valid])
    vectorized_test = np.array([datapoint["tokens"] for datapoint in test])
    X_train, y_train, X_test, y_test = [], [], [], []

    for i in range(len(vectorized_train)):
        vectorized_doc = vectorized_train[i]
        for j in range(len(vectorized_doc)):
            context = " ".join(vectorized_doc[max(0, j - 2): min(len(vectorized_doc) - 1, j + 2)])
            embedded_context = embed(context, word2vec_model)
            # import pdb
            # pdb.set_trace()
            label = train_valid[i]["ner_ids"][j]
            X_train.append(embedded_context)
            y_train.append(label)

    for i in range(len(vectorized_test)):
        vectorized_doc = vectorized_test[i]
        for j in range(len(vectorized_doc)):
            context = " ".join(vectorized_doc[max(0, j - 2): min(len(vectorized_doc) - 1, j + 2)])
            embedded_context = embed(context, word2vec_model)
            label = test[i]["ner_ids"][j]
            X_test.append(embedded_context)
            y_test.append(label)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def classifier_experiment(X_train, y_train, X_test, y_test):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    train_classifier_head(X_train, y_train, X_test, y_test)
    ensemble_voting(X_train, y_train, X_test, y_test)


def main():
    prepare_data()
    docs = load_liro_dataset()
    train_word2vec(docs)
    word2vec_model = load_word2vec()
    X_train, y_train, X_test, y_test = create_train_test_data(word2vec_model)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[-1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[-1]))

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    classifier_experiment(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
