#!/usr/bin/env python


import math
import glob
import random

from Dictionary import Dictionary
from BatchManager import BatchManager


def load_data(data_file, is_train=1, delimiter="\t", field_delimiter="|"):
    data = list()
    tmp_flag = True
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if tmp_flag:
                tmp_flag = False
                continue

            tokens = line.split(delimiter)
            prefix = tokens[0]
            title = tokens[1]
            tag = tokens[2]
            target = tokens[3]
            flag = tokens[4]
            raw_texts = tokens[6:16]
            scores = tokens[16:26]

            texts = list()
            segments = list()
            for text in raw_texts:
                segment = text.split(field_delimiter)
                segments.append(segment)
                texts.append("".join(segment))
            prefix = prefix.split(field_delimiter)
            title = title.split(field_delimiter)
            sample = {
                "prefix": prefix,
                "title": title,
                "tag": tag,
                "flag": flag,
                "segments": segments, 
                "texts": texts,
                "scores": [float(item) for item in scores],
                "targets": int(target),
                "line": line.replace("\t", "  "),
            }

            data.append(sample)
    if is_train:
        train_set = [item for item in data if item["flag"] == "train"]
        valid_set = [item for item in data if item["flag"] == "valid"]
        return train_set, valid_set
    else:
        predict_set = [item for item in data if item["flag"] == "test"]
        return predict_set

def data_batch(data, params, dictionary_path=None):
    cutoff = params["vocab_cutoff"]
    if dictionary_path is None:
        dictionary = Dictionary(data, cutoff=cutoff)
    else:
        dictionary = Dictionary()
        dictionary.load(data)
    pad_id = dictionary.pad_id()
    batch_manager = BatchManager(data, params, pad_id)
    
    return batch_manager



