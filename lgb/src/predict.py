#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import os
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

sys.path.append("conf")
import config
import logging

logging.basicConfig(
        level = logging.INFO,
        format = "[%(asctime)s] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        )

def f1_score_metric(pred, d_valid):
    label = d_valid.get_label()
    pred = [int(i>=0.4) for i in pred]

    return "f1_score", f1_score(label, pred), True


def main():
    logging.info("load data ...")
    data = pd.read_csv(config.MODEL_DATA_FILE, sep="\t", encoding="utf-8", low_memory=False)
    # encode categorical cols
    for col in config.CATEGORICAL_COLS:
        data[col] = data[col].astype("str")
        data[col] = LabelEncoder().fit_transform(data[col])

    train_data = data[data["flag"] == "train"]
    valid_data = data[data["flag"] == "valid"]
    test_data = data[data["flag"] == "test"]

    train_data = pd.concat([train_data, valid_data])

    X = np.array(train_data.drop(config.IGNORE_COLS, axis = 1))
    y = np.array(train_data['label'])
    X_test = np.array(test_data.drop(config.IGNORE_COLS, axis = 1))
    print ("X.shape: ", X.shape)
    print ("y.shape: ", y.shape)


    logging.info("train model ...")
    result_logloss = []
    result_submit = []
    N = 5
    skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 32,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1, 
    }
    
    for k, (train_in, test_in) in enumerate(skf.split(X, y)):
        print ("============================= train _K_ flod", k, "====================================")
        X_train, X_valid, y_train, y_valid = X[train_in], X[test_in], y[train_in], y[test_in]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                #valid_sets=[lgb_train, lgb_eval],
                valid_sets=lgb_eval,
                early_stopping_rounds=50,
                verbose_eval=10,
                feval=f1_score_metric
                )
        
        valid_f1_score = f1_score(y_valid, np.where(gbm.predict(X_valid, num_iteration=gbm.best_iteration) > 0.4, 1,0))
        print ("best_iteration: ", gbm.best_iteration)
        print ("valid_f1_score: ", valid_f1_score)

        result_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
        result_submit.append(gbm.predict(X_test, num_iteration=gbm.best_iteration))

    print ("train_logloss: ", np.mean(result_logloss))
    result = test_data.copy()
    result['label'] = list(np.sum(np.array(result_submit), axis=0) / N)
    result['label'] = result['label'].apply(lambda x: round(x))
    print ("test_logloss: ", np.mean(result.label))

    feature_importances = sorted(zip(train_data.drop(config.IGNORE_COLS, axis = 1).columns, gbm.feature_importance()), key=lambda x:x[1])
    print (feature_importances)

    time = sys.argv[1]
    result['label'].to_csv("./output/result_lgb_"+time+".csv", index=False, encoding='utf-8', header=None)
    
    logging.info("done ...")

if __name__ == "__main__":
    main()
