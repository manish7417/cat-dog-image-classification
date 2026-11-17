import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test1')
CAT_LBL = 0
DOG_LBL = 1
CAT = 'cat'
DOG = 'dog'
LABEL_MAP = {
    CAT: CAT_LBL,
    DOG: DOG_LBL
}
DATA_SIZE = 18_000
IMG_SIZE = 110
SPLIT_RATIO = 0.8