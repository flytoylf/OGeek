#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: yangxiaohan@xiaomi.com
# @Date: 2018-09-30
#

import re
import sys
import json
import time
import logging
import pandas as pd
import numpy as np

sys.path.append("conf")
import config
sys.path.append("tool")
import nlp_basic

logging.basicConfig(
        level = logging.INFO,
        format = "[%(asctime)s] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        )

all_segments = set()

def get_segment(text, tag):
    segs = list(nlp_basic.segment(text))
    if tag == "train":
        for seg in segs:
            if seg not in all_segments:
                all_segments.add(seg)
    return "|".join(segs)


def move_useless_char(s):
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+??！，。？?、~@#￥%……&*（）]+", "", s)


def parse_query_prediction(query_prediction, data_tag):
    if query_prediction == "":
        return 0, "\t".join(["PAD"] * 10), "\t".join(['0.000'] * 10)

    json_data = json.loads(query_prediction)
    result = sorted(json_data.items(), key=lambda d:d[1], reverse = True)
    texts = [get_segment(move_useless_char(item[0]), data_tag) for item in result]
    scores = [item[1] for item in result]

    n = len(texts)
    return n, "\t".join(texts + ["PAD"]*(10-n)), "\t".join(scores + ['0.000']*(10-n))


def load_data(input_file, output_file, data_tag):
    with open(output_file, 'w') as fo:
        fo.write("\t".join([
            "prefix", "title", "tag", "label", "flag", "num",
            "\t".join(["text_"+str(i+1) for i in range(10)]), 
            "\t".join(["score_"+str(i+1) for i in range(10)])
            ]))
        fo.write("\n")

        with open(input_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if data_tag == "test":
                    prefix, query_prediction, title, tag = line.split("\t")
                    label = "-1"
                else:
                    prefix, query_prediction, title, tag, label = line.split("\t")

                prefix = move_useless_char(prefix)
                title = move_useless_char(title)

                n, prediction_text, prediction_score = parse_query_prediction(query_prediction, data_tag)
                fo.write("\t".join([
                    get_segment(prefix, data_tag), get_segment(title, data_tag), tag, label, data_tag, str(n), prediction_text, prediction_score
                    ]))
                fo.write("\n")


def get_word_vector():
    logging.info("load word vectors ...")
    with open(config.WORD_VECTORS_FILE, 'w') as fo:
        with open(config.ORI_WORD_VECTORS_FILE, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                word = tokens[0]
                if word in all_segments:
                    fo.write(line)
                    fo.write("\n")
               

def merge_data():
    train_data = pd.read_csv(config.TRAIN_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    valid_data = pd.read_csv(config.VALID_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    test_data = pd.read_csv(config.cTEST_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)


def main():
    data_type = "test_A"
    
    time_point = time.time()
    if data_type == "test_A":
        logging.info("load train data ...")
        load_data(config.ORI_TRAIN_DATA_FILE, config.TRAIN_DATA_FILE, "train")
        logging.info("load train data cost time: "+str(time.time()-time_point))

        logging.info("load valid data ...")
        time_point = time.time()
        load_data(config.ORI_VALID_DATA_FILE, config.VALID_DATA_FILE, "valid")
        logging.info("load valid data cost time: "+str(time.time()-time_point))

        logging.info("load test data ...")
        time_point = time.time()
        load_data(config.ORI_TEST_DATA_FILE, config.TEST_DATA_FILE, "test")
        logging.info("load test data cost time: "+str(time.time()-time_point))

        logging.info("get word vector ...")
        time_point = time.time()
        get_word_vector()
        logging.info("get word vector cost time: "+str(time.time()-time_point))

    elif data_type == "test_B":
        logging.info("load test B data ...")
        time_point = time.time()
        load_data(config.ORI_TEST_B_DATA_FILE, config.TEST_B_DATA_FILE, "test")
        logging.info("load test B data cost time: "+str(time.time()-time_point))

    else:
        logging.error("wrong data_type!")

    train_data = pd.read_csv(config.TRAIN_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    valid_data = pd.read_csv(config.VALID_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    if data_type == "test_A":
        test_data = pd.read_csv(config.TEST_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    elif data_type == "test_B":
        test_data = pd.read_csv(config.TEST_B_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    else:
        logging.error("wrong data_type!")

    data = pd.concat([train_data, valid_data, test_data])
    data.to_csv(config.ORI_DATA_FILE, sep="\t", index=False, encoding="utf-8")
                    
    logging.info("done!")


if __name__ == "__main__":
    main()
