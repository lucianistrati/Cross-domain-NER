from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from used_repos.personal.aggregated_personal_repos.Cross_domain_NER.src.common.util import get_all_data
from sklearn.metrics import f1_score
from statistics import mean
from tqdm import tqdm

import numpy as np

import spacy
import pdb


nlp = spacy.load("en_core_web_sm")


def finetune_spacy_engine():
    """

    :return:
    """
    pass


def vectorize(cur_labels, tag_to_id, spacy_to_dataset_labels):
    """

    :param cur_labels:
    :return:
    """
    new_cur_labels = []
    # print(cur_labels)
    for i in range(len(cur_labels)):
        if cur_labels[i][0] == "O":
            continue
        label = cur_labels[i][0]
        if label in ["ORGANIZATION", "NUMERIC_VALUE"]:
            label = spacy_to_dataset_labels[label]
        new_cur_labels.append((tag_to_id[label], cur_labels[i][1], cur_labels[i][2]))
    return new_cur_labels


def set_gt_dict(cur_gt_dict, cur_gt_labels):
    """

    :param cur_gt_labels:
    :return:
    """
    for cur_gt_label in cur_gt_labels:
        s, e = cur_gt_label[1], cur_gt_label[2]
        lbl = cur_gt_label[0]
        for i in range(s, e + 1):
            cur_gt_dict[i] = lbl
    return cur_gt_dict


def set_pred_dict(cur_pred_dict, cur_pred_labels):
    """

    :param cur_pred_labels:
    :return:
    """
    for cur_pred_label in cur_pred_labels:
        s, e = cur_pred_label[1], cur_pred_label[2]
        lbl = cur_pred_label[0]
        for i in range(s, e + 1):
            cur_pred_dict[i] = lbl
    return cur_pred_dict


def main():
    data, tag_to_id = get_all_data(change_ner_tags=True, change_ner_ids=True)
    test = data["test"]

    scores = []
    spacy_to_dataset_labels = {"NUMERIC_VALUE": "NUMERIC", "ORGANIZATION": "ORG"}

    cur_pred_dict = dict()
    cur_gt_dict = dict()

    for doc in tqdm(test):
        spacy_doc = nlp(doc["reconstructed_document"])
        # vec = []
        start_chars_gt = doc["start_char"]
        end_chars_gt = doc["end_char"]
        cur_gt_labels = list(zip(doc["ner_tags"], start_chars_gt, end_chars_gt))
        cur_pred_labels = []
        for ent in spacy_doc.ents:
            if ent.label_ == "PRODUCT":
                continue
            spacy_label = ent.label_
            if ent.label in ["ORGANIZATION", "NUMERIC_VALUE"]:
                spacy_label = spacy_to_dataset_labels[spacy_label]

            cur_pred_labels.append((spacy_label, ent.start_char, ent.end_char))

        cur_gt_labels = vectorize(cur_gt_labels, tag_to_id, spacy_to_dataset_labels)
        cur_pred_labels = vectorize(cur_pred_labels, tag_to_id, spacy_to_dataset_labels)

        cur_pred_dict = set_pred_dict(cur_pred_dict, cur_pred_labels)
        cur_gt_dict = set_gt_dict(cur_gt_dict, cur_gt_labels)

        # print(len(cur_gt_dict), len(cur_pred_dict))

        for key in cur_gt_dict.keys():
            if key not in cur_pred_dict.keys():
                if key >= len(doc["reconstructed_document"]):
                    continue
                if doc["reconstructed_document"][key] != " ":
                    cur_pred_dict[key] = tag_to_id["O"]

        for key in cur_pred_dict.keys():
            if key not in cur_gt_dict.keys():
                if key >= len(doc["reconstructed_document"]):
                    continue
                if doc["reconstructed_document"][key] != " ":
                    cur_gt_dict[key] = tag_to_id["O"]

        common_keys = set(list(cur_pred_dict.keys())).intersection(set(list(cur_gt_dict.keys())))
        cur_gt_dict = {common_key: cur_gt_dict[common_key] for common_key in common_keys}
        cur_pred_dict = {common_key: cur_pred_dict[common_key] for common_key in common_keys}

        cur_gt_dict = sorted(cur_gt_dict.items())
        cur_pred_dict = sorted(cur_pred_dict.items())

        final_gt_labels = [elem[1] for elem in cur_gt_dict]
        final_pred_labels = [elem[1] for elem in cur_pred_dict]

        scores.append(f1_score(final_pred_labels, final_gt_labels, average="weighted"))

    print(mean(scores))


if __name__ == "__main__":
    main()
