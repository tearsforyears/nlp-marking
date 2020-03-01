# coding=utf-8
import os

# database and project root dir
ROOT_DIR = os.path.dirname(__file__) + '/'
DATA_ROOT = ROOT_DIR + r'data/'
ORIGIN_COMMENT = DATA_ROOT + r'douban_data.txt'
COLLECTION_NAME = 'douban3'

# training data-set
TRAIN_DATA = DATA_ROOT + r'iter1/train.npy'
TRAIN_LABELS = DATA_ROOT + r'iter1/labels.npy'
DATA_SIZE = 106001
DATA_SET_CACHE_DIR = DATA_ROOT + r'data_set_cache/'
SEQUENCE_MAX_LENGTH = 30
LABELS_CAREGORIES = 5
# test-set generate
SPLIT_RATE = 0.9

# word2vec (CBOW) model for transfer learning
WORD2VEC_MODEL_PATH = r"F:/note/models/word2vec/baike_26g_news_13g_novel_229g.bin"
EMBEDDING_UNITS = 128

# stop words
STOP_WORDS_PATH = DATA_ROOT + r'stopwords.dat'

# model parameters
GRU_HIDDEN_UNITS = 64
GRU_DROPOUT = 0.3
DENSE_HIDDEN_UNITS = 5
DENSE_DROPOUT = 0.3
REGULAR_NAMDA = 0.0001

# train parameters
LEARNING_RATE = 1e-4
DECAY_RATE = 1e-4
BATCH_SIZE = 64
EPOCH = 100
STEPS = 10000000
MODEL_FILE_PATH = DATA_ROOT + r'iter1/marking-model.h5'
LAST_MODEL_NAME = DATA_ROOT + r'iter1/marking-model-final.h5'

# TensorBoard logdir
LOG_DIR = ROOT_DIR + r'visualize/'

# predicting
SCORE_LIST = ['一般', "积极", "非常积极", "非常消极", "消极"]
SCORE_LIST2 = ['积极', '消极']
SCORE_LIST3 = ['消极', '一般', '积极']
