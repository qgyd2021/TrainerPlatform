#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import pickle

import graphviz
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
    parser.add_argument("--graphviz_filename", default="clf.graphviz", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    file_dir = Path(args.file_dir)

    model_filename = file_dir / args.model_filename
    graphviz_filename = file_dir / args.graphviz_filename

    with open(model_filename, "rb") as f:
        clf = pickle.load(f)

    # 可视化决策树
    feature_names = [
        "mean",
        "var",

        # "per1",
        # "per25",
        # "per50",
        # "per75",
        # "per99",

        # "silence_rate",
        # "mean_non_silence",
        # "silence_count",
        # "var_var_non_silence",
        # "var_non_silence",
        # "var_non_silence_rate",
        # "var_var_whole",

    ]
    classes = ["non_voice", "voice"]

    dot_data = export_graphviz(
        clf,
        class_names=classes,
        feature_names=feature_names,
        out_file=None
    )
    graph = graphviz.Source(dot_data)

    graph.render(graphviz_filename.as_posix())

    return


if __name__ == '__main__':
    main()
