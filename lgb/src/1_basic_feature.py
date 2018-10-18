#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import os
import time
import logging
import numpy as np
import pandas as pd

from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

sys.path.append("conf")
import config

logging.basicConfig(
        level = logging.INFO,
        format = "[%(asctime)s] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        )


logging.info("load stop words ...")
stop_words = set()   # stopwords
with open(config.STOP_WORDS_FILE, 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        stop_words.add(line)


logging.info("load word vectors ...")
word2vec = dict()
text2vec = dict()
with open(config.WORD_VECTORS_FILE, 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        word = tokens[0]
        vecs = tokens[1:]
        word2vec[word] = [float(vec) for vec in vecs]


def get_distance(q, t, distance_func):
    if q == "PAD" or t == "PAD":
        return -1
    flag_q, q_vector = text2vec[q]
    flag_t, t_vector = text2vec[t]

    if flag_q and flag_t:
        return distance_func(q_vector, t_vector)
    else:
        return -1


def get_skew(text):
    if text == "PAD":
        return -1

    flag, vector = text2vec[text]
    if flag:
        return skew(vector)
    else:
        return -1


def get_kurtosis(text):
    if text == "PAD":
        return -1

    flag, vector = text2vec[text]
    if flag:
        return kurtosis(vector)
    else:
        return -1


def get_pointwise_feature(data, col):
    data[col+"_len"] = data[col].apply(lambda x: len(x.replace("|", "")) if x != "PAD" else -1)
    data[col+"_uniq_len"] = data[col].apply(lambda x: len("".join(set(x.replace("|", "")))) if x != "PAD" else -1)
    data[col+"_seg_len"] = data[col].apply(lambda x: len(x.replace("|", "")) if x != "PAD" else -1)
    data[col+"_skew"] = data[col].apply(lambda x:get_skew(x))
    data[col+"_kurtosis"] = data[col].apply(lambda x:get_kurtosis(x))
    
    return data


def get_pairwise_feature(data, q, t):
    data[q+"_"+t+"_diff_len"] = data[[t+"_len", q+"_len"]].apply(lambda x: x[t+"_len"] - x[q+"_len"] if (x[t+"_len"]!=-1 and x[q+"_len"]!=-1) else -10, axis=1)
    data[q+"_"+t+"_common_seg"] = data.apply(lambda x: len(set(x[q].lower().split("|")).intersection(set(x[t].lower().split("|"))) if x[q]!="PAD" and x[t]!="PAD" else []), axis=1)

    for distance_func in [cosine, jaccard, cityblock, canberra, euclidean, minkowski, braycurtis]:
        data[q+"_"+t+"_"+distance_func.__name__+"_distance"] = data[[q, t]].apply(lambda x:get_distance(x[q], x[t], distance_func), axis=1)

    return data


def get_text_vector(text, text2vec):
    if text in text2vec:
        return
    words = text.split('|')
    words = [w for w in words if not w in stop_words]
    M = []
    for w in words:
        if w in word2vec:
            M.append(word2vec[w])
        else:
            continue

    if not M:
        text2vec[text] = (0, [0.] * 100)
        return

    M = np.array(M)
    v = M.sum(axis=0)
    text2vec[text] = (1, v / np.sqrt((v ** 2).sum()))
    return
    

def main():
    logging.info("load data ...")
    data = pd.read_csv(config.ORI_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)

    logging.info("get text vectors ...")
    for col in ["prefix", "title", "text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10"]:
        data[col] = data[col].astype(str)
        data[col].apply(lambda x: get_text_vector(x, text2vec))
    
    logging.info("get feature ... ") 
    logging.info("\tpointwise feature ...")
    for col in ["prefix", "title", "text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10"]:
        logging.info("\tcol: "+col)
        data = get_pointwise_feature(data, col)

    logging.info("\tpairwise feature ...")
    for col in ["prefix", "text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10"]:
        logging.info("\tcol: "+col)
        data = get_pairwise_feature(data, col, "title")

    logging.info("save data ...")
    data.to_csv(config.BASE_FEATURE_FILE, sep="\t", index=False, encoding="utf-8")
    
    logging.info("done ...")


if __name__ == "__main__":
    main()


