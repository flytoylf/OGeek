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


def get_ctr_feature(cols, data, train_data):
    for col in cols:
        tmp = train_data.groupby(col, as_index=False)["label"].agg({col+"_click": "sum", col+"_show": "count"})
        tmp[col+"_ctr"] = tmp[col+"_click"] / (tmp[col+"_show"] + 3)
        for tmp_col in [col+"_show", col+"_click", col+"_ctr"]:
            tmp[tmp_col] = tmp[tmp_col].apply(lambda x: x if x != "PAD" else -1)

        data = pd.merge(data, tmp, on=col, how="left")

    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            group_col = [cols[i], cols[j]]
            tmp = train_data.groupby(group_col, as_index=False)["label"].agg({"_".join(group_col)+"_click": "sum", "_".join(group_col)+"_show": "count"})
            tmp["_".join(group_col)+"_ctr"] = tmp["_".join(group_col)+"_click"] / (tmp["_".join(group_col)+"_show"] + 3)
            for tmp_col in ["_".join(group_col)+"_show", "_".join(group_col)+"_click", "_".join(group_col)+"_ctr"]:
                tmp[tmp_col] = tmp[group_col+[tmp_col]].apply(lambda x: x[tmp_col] if "PAD" not in x[group_col].values else -1, axis=1)
            data = pd.merge(data, tmp, on=group_col, how="left")
    
    return data


def main():
    logging.info("load data ...")
    data = pd.read_csv(config.ORI_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)

    train_data = data[data["flag"] == "train"]
    valid_data = data[data["flag"] == "valid"]
    test_data = data[data["flag"] == "test"]
    
    # base str
    logging.info("base str ...")
    cols = ["prefix", "title", "tag"]
    data = get_ctr_feature(cols, data, train_data)

    group_col = cols
    tmp = train_data.groupby(group_col, as_index=False)["label"].agg({"_".join(group_col)+"_click": "sum", "_".join(group_col)+"_show": "count"})
    tmp["_".join(group_col)+"_ctr"] = tmp["_".join(group_col)+"_click"] / (tmp["_".join(group_col)+"_show"] + 3)
    data = pd.merge(data, tmp, on=cols, how="left")

    # query_prediction ctr
    logging.info("query_prediction str ...")
    cols = ["text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10"]
    data = get_ctr_feature(cols, data, train_data)

    data = data.fillna(-1)

    logging.info("save data ...")
    data.to_csv(config.STATISTICS_FEATURE_FILE, sep="\t", index=False, encoding="utf-8")
    
    logging.info("done ...")
if __name__ == "__main__":
    main()
