from used_repos.personal.aggregated_personal_repos.Cross_domain_NER.src.common.util import get_data, get_all_data
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
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


def document_preprocess(document: str):
    """

    :param document:
    :return:
    """
    return word_tokenize(document)


def ensemble_voting(X_train: np.ndarrray, y_train: np.ndarrray, X_test: np.ndarrray, y_test: np.ndarrray):
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


def train_classifier_head(X_train: np.ndarrray, y_train: np.ndarrray, X_test: np.ndarrray, y_test: np.ndarrray):
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


def embed(text: str, word2vec_model: gensim.models.word2vec.Word2Vec):
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


def create_train_test_data(data: Dict, word2vec_model: gensim.models.word2vec.Word2Vec):
    """

    :param data:
    :param word2vec_model:
    :return:
    """
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


def classifier_experiment(X_train: np.ndarrray, y_train: np.ndarrray, X_test: np.ndarrray, y_test: np.ndarrray):
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
    data, _ = get_all_data(first_n=100)
    X_train, y_train, X_test, y_test = create_train_test_data(data)

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    classifier_experiment(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
