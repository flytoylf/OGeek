# data file
ORI_TRAIN_DATA_FILE = "./data/download/oppo_round1_train_20180929.txt"
ORI_VALID_DATA_FILE = "./data/download/oppo_round1_vali_20180929.txt"
ORI_TEST_DATA_FILE = "./data/download/oppo_round1_test_A_20180929.txt"
ORI_TEST_B_DATA_FILE = "./data/download/oppo_round1_test_A_20180929.txt"
TRAIN_DATA_FILE = "./data/train.txt"
VALID_DATA_FILE = "./data/valid.txt"
TEST_DATA_FILE = "./data/test.txt"

STOP_WORDS_FILE = "./data/stop_words.txt"
ORI_WORD_VECTORS_FILE = "./data/baike.vec"
WORD_VECTORS_FILE = "./data/word_vectors.txt"

ORI_DATA_FILE = "./data/0_ori_data.txt"
BASE_FEATURE_FILE = "./data/1_base_feature.txt"
PAIR_FEATURE_FILE = "./data/2_pair_feature.txt"
STATISTICS_FEATURE_FILE = "./data/3_statistics_feature.txt"

SAMPLE_ORI_DATA_FILE = "./data/sample_0_ori_data.txt"
SAMPLE_BASE_FEATURE_FILE = "./data/sample_base_feature.txt"
SAMPLE_PAIR_FEATURE_FILE = "./data/sample_pair_feature.txt"

MODEL_DATA_FILE = "./data/model_feature.txt"
SAMPLE_MODEL_DATA_FILE = "./data/sample_base_feature.txt"


# col config
BASE_COLS = [
        "prefix", "title", "tag", "label", "flag", "num", "text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10", "score_1", "score_2", "score_3", "score_4", "score_5", "score_6", "score_7", "score_8", "score_9", "score_10"
        ]

CATEGORICAL_COLS = [
        "tag",
        ]


NUMERIC_COLS = [
        ]


TEXT_COLS = [
        ]


IGNORE_COLS = [
        "prefix", "title", "label", "flag", "text_1", "text_2", "text_3", "text_4", "text_5", "text_6", "text_7", "text_8", "text_9", "text_10",
        ]
