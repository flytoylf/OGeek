#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.append("conf")
import config
import logging 

logging.basicConfig(
        level = logging.INFO,
        format = "[%(asctime)s] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        )

def main():
    logging.info("load data ...")
    base_feature = pd.read_csv(config.BASE_FEATURE_FILE, sep="\t", encoding="utf-8", low_memory=False)
    pair_feature = pd.read_csv(config.PAIR_FEATURE_FILE, sep="\t", encoding="utf-8", low_memory=False)
    pair_feature = pair_feature.drop(BASE_COLS, axis=1)
    statistics_feature = pd.read_csv(config.STATISTICS_FEATURE_FILE, sep="\t", encoding="utf-8", low_memory=False)
    statistics_feature = statistics_feature.drop(BASE_COLS, axis=1)
    data = pd.concat([base_feature, pair_feature, statistics_feature], axis = 1)

    # encode categorical cols
    for col in config.CATEGORICAL_COLS:
        data[col] = data[col].astype("str")
        data[col] = LabelEncoder().fit_transform(data[col])


    logging.info("save data ...")
    data.to_csv(config.MODEL_DATA_FILE, sep="\t", index=False, encoding="utf-8")

    logging.info("done ...")

if __name__ == "__main__":
    main()
