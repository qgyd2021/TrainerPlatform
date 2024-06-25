#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import model_selection
from sklearn.utils.validation import check_is_fitted


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", default="./file_dir", type=str)
    parser.add_argument("--train_dataset", default="train.xlsx", type=str)
    parser.add_argument("--test_dataset", default="test.xlsx", type=str)

    parser.add_argument("--model_filename", default="clf.pkl", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    file_dir = Path(args.file_dir)
    train_dataset_file = file_dir / args.train_dataset
    valid_dataset_file = file_dir / args.test_dataset

    model_filename = file_dir / args.model_filename

    train_df = pd.read_excel(train_dataset_file)
    valid_df = pd.read_excel(valid_dataset_file)

    classes = [
        "mean",
        "var",

        "per1",
        "per25",
        "per50",
        "per75",
        "per99",

        "silence_rate",
        "mean_non_silence",
        "silence_count",
        "var_var_non_silence",
        "var_non_silence",
        "var_non_silence_rate",
        "var_var_whole",

    ]

    x = list()
    y = list()
    for i, row in train_df.iterrows():
        features = row["features"]
        features = json.loads(features)

        x_ = [
            features["mean"],
            features["var"],

            # features["per1"],
            # features["per25"],
            # features["per50"],
            # features["per75"],
            # features["per99"],

            # features["silence_rate"],
            # features["mean_non_silence"],
            # features["silence_count"],
            # features["var_var_non_silence"],
            # features["var_non_silence"],
            # features["var_non_silence_rate"],
            # features["var_var_whole"],

        ]
        y_ = row["label"]
        x.append(x_)
        y.append(y_)

    clf = DecisionTreeClassifier(
        min_samples_split=2000,
        min_samples_leaf=1000,
        random_state=0,
        class_weight={
            "non_voice": 1,
            "voice": 5,
        }
    )
    clf.fit(x, y)
    print("clf.classes_: {}".format(clf.classes_))

    train_accuracy = clf.score(x, y)

    x = list()
    y = list()
    for i, row in valid_df.iterrows():
        features = row["features"]
        features = json.loads(features)

        x_ = [
            features["mean"],
            features["var"],

            # features["per1"],
            # features["per25"],
            # features["per50"],
            # features["per75"],
            # features["per99"],

            # features["silence_rate"],
            # features["mean_non_silence"],
            # features["silence_count"],
            # features["var_var_non_silence"],
            # features["var_non_silence"],
            # features["var_non_silence_rate"],
            # features["var_var_whole"],

        ]
        y_ = row["label"]
        x.append(x_)
        y.append(y_)

    valid_accuracy = clf.score(x, y)

    with open(model_filename, "wb") as f:
        pickle.dump(clf, f)

    print("train_accuracy: {}, valid_accuracy: {}".format(train_accuracy, valid_accuracy))
    return


if __name__ == '__main__':
    main()
