#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import os
import time
import logging
import numpy as np
import pandas as pd

from collections import defaultdict
from collections import Counter

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


def get_weight(count, eps=100, min_count=2):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)


def word_match_share(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    q_words = set()
    t_words = set()
    for word in q.split('|'):
        if word not in stop_words:
            q_words.add(word)
    for word in t.split('|'):
        if word not in stop_words:
            t_words.add(word)
    if len(q_words) == 0 or len(t_words) == 0:
        return 0.

    shared_words_in_q = [w for w in q_words if w in t_words]
    shared_words_in_t = [w for w in t_words if w in q_words]
    R = float(len(shared_words_in_q) + len(shared_words_in_t))/(len(q_words) + len(t_words))
    return R

def tfidf_word_match_share(q, t, weights):
    if q == "PAD" or t == "PAD":
        return -1

    q_words = set()
    t_words = set()
    for word in q.split('|'):
        if word not in stop_words:
            q_words.add(word)
    for word in t.split('|'):
        if word not in stop_words:
            t_words.add(word)

    if len(q_words) == 0 or len(t_words) == 0:
        return 0.

    shared_weights = [weights.get(w, 0) for w in q_words if w in t_words] + [weights.get(w, 0) for w in t_words if w in q_words]
    total_weights = [weights.get(w, 0) for w in q_words] + [weights.get(w, 0) for w in t_words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def tfidf_word_match_share_stops(q, t, weights):
    if q == "PAD" or t == "PAD":
        return -1

    q_words = set()
    t_words = set()
    for word in q.split('|'):
        q_words.add(word)
    for word in t.split('|'):
        t_words.add(word)
    if len(q_words) == 0 or len(t_words) == 0:
        return 0.

    shared_weights = [weights.get(w, 0) for w in q_words if w in t_words] + [weights.get(w, 0) for w in t_words if w in q_words]
    total_weights = [weights.get(w, 0) for w in q_words] + [weights.get(w, 0) for w in t_words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

def jaccard(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    wic = set(q.split('|')).intersection(set(t.split('|')))
    uw = set(q.split('|')).union(t.split('|'))
    if len(uw) == 0:
        uw = [1]
    return float(len(wic) / len(uw))

def wc_diff(q, t):
    if q == "PAD" or t == "PAD":
        return -10

    return len(q.split('|')) - len(t.split('|'))

def wc_ratio(q, t):
    if q == "PAD" or t == "PAD":
        return -1
    
    l1 = len(q.split('|')) * 1.0
    l2 = len(t.split('|')) * 1.0
    if l2 == 0:
        return -1

    return l1 / l2

def wc_diff_unique(q, t):
    if q == "PAD" or t == "PAD":
        return -10

    return len(set(q.split('|'))) - len(set(t.split('|')))

def wc_ratio_unique(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    l1 = len(set(q.split('|'))) * 1.0
    l2 = len(set(t.split('|'))) * 1.0
    if l2 == 0:
        return -1

    return l1 / l2

def wc_diff_unique_stop(q, t):
    if q == "PAD" or t == "PAD":
        return -10

    return len([x for x in set(q.split('|')) if x not in stop_words]) - len([x for x in set(t.split('|')) if x not in stop_words])

def wc_ratio_unique_stop(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    l1 = len([x for x in set(q.split('|')) if x not in stop_words]) * 1.0
    l2 = len([x for x in set(t.split('|')) if x not in stop_words]) * 1.0
    if l2 == 0:
        return -1

    return l1 / l2

def same_start(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    if not q or not t:
        return 0
    qs = q.split('|')
    ts = t.split('|')
    l = min(len(qs), len(ts))
    result = 0
    for i in range(l):
        if qs[i] == ts[i]:
            result += len(qs[i])

    return result

def char_diff(q, t):
    if q == "PAD" or t == "PAD":
        return -10

    return len(''.join(q.split('|'))) - len(''.join(t.split('|')))

def char_diff_unique_stop(q, t):
    if q == "PAD" or t == "PAD":
        return -10

    return len(''.join([x for x in set(q.split('|')) if x not in stop_words])) - len(''.join([x for x in set(t.split('|')) if x not in stop_words]))

def total_unique_words(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    return len(set(q.split('|')).union(t.split('|')))

def total_unq_words_stop(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    return len([x for x in set(q.split('|')).union(t.split('|')) if x not in stop_words])

def char_ratio(q, t):
    if q == "PAD" or t == "PAD":
        return -1

    l1 = len(''.join(q.split('|'))) * 1.0
    l2 = len(''.join(t.split('|'))) * 1.0
    if l2 == 0:
        return -1

    return l1 / l2

def get_pairwise_feature(data, q, t, weights):
    result = pd.DataFrame()
    data[q+"_"+t+"_"+"word_match_share"] = data[[q, t]].apply(lambda x:word_match_share(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"tfidf_word_match_share"] = data[[q, t]].apply(lambda x:tfidf_word_match_share(x[q], x[t], weights), axis=1)
    data[q+"_"+t+"_"+"tfidf_word_match_share_stops"] = data[[q, t]].apply(lambda x:tfidf_word_match_share_stops(x[q], x[t], weights), axis=1)
    
    data[q+"_"+t+"_"+"jaccard"] = data[[q, t]].apply(lambda x:jaccard(x[q], x[t]), axis=1)
#    data[q+"_"+t+"_"+"wc_diff"] = data[[q, t]].apply(lambda x:wc_diff(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"wc_ratio"] = data[[q, t]].apply(lambda x:wc_ratio(x[q], x[t]), axis=1)
#    data[q+"_"+t+"_"+"wc_diff_unique"] = data[[q, t]].apply(lambda x:wc_diff_unique(x[q], x[t]), axis=1)
#    data[q+"_"+t+"_"+"wc_diff_unique_stop"] = data[[q, t]].apply(lambda x:wc_diff_unique_stop(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"wc_ratio_unique"] = data[[q, t]].apply(lambda x:wc_ratio_unique(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"wc_ratio_unique_stop"] = data[[q, t]].apply(lambda x:wc_ratio_unique_stop(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"same_start"] = data[[q, t]].apply(lambda x:same_start(x[q], x[t]), axis=1)
    
    data[q+"_"+t+"_"+"char_diff"] = data[[q, t]].apply(lambda x:char_diff(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"char_diff_unique_stop"] = data[[q, t]].apply(lambda x:char_diff_unique_stop(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"total_unique_words"] = data[[q, t]].apply(lambda x:total_unique_words(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"total_unq_words_stop"] = data[[q, t]].apply(lambda x:total_unq_words_stop(x[q], x[t]), axis=1)
    data[q+"_"+t+"_"+"char_ratio"] = data[[q, t]].apply(lambda x:char_ratio(x[q], x[t]), axis=1)

    return data

def main():
    logging.info("load data ...")
    data = pd.read_csv(config.ORI_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)

    logging.info("get total words ...")
    total_words = list()
    for col in ["prefix", "title", "text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10"]:
        data[col] = data[col].astype(str)
        for i in range(data.shape[0]):
            total_words += data[col][i].split('|')

    counts = Counter(total_words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    logging.info("\tpairwise feature ...")
    for col in ["prefix", "text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10"]:
        logging.info("\tcol: "+col)
        data = get_pairwise_feature(data, col, "title", weights)
    

    logging.info("save data ...")
    data.to_csv(config.PAIR_FEATURE_FILE, sep="\t", index=False, encoding="utf-8")
    
    logging.info("done ...")

if __name__ == "__main__":
    main()
