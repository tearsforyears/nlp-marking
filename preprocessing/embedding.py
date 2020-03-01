# coding=utf-8
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from preprocessing.getdata import get_words_list
import numpy as np
from settings import *
import os


def load_embedding_model(path=WORD2VEC_MODEL_PATH):
    return KeyedVectors.load_word2vec_format(path, binary=True)


def load_data(train_data=TRAIN_DATA, train_labels=TRAIN_LABELS):
    '''
    Note:
        load the data using .npy files
    '''
    if os.path.exists(train_data) and os.path.exists(train_labels):
        return np.load(train_data), np.load(train_labels)
    else:
        raise RuntimeError('[error]: the training data had not found or not been generated')


def get_label_distribution(begin=0, end=DATA_SIZE):
    labels = np.load(DATA_SET_CACHE_DIR + 'label.npy')
    count = [0, 0, 0, 0, 0]
    for label in labels[begin:end]:
        count[label + 2] += 1
    print(count)
    # marking value distribution = [6890, 11377, 32191, 34919, 20624]


def load_split_data(train_data=TRAIN_DATA, train_labels=TRAIN_LABELS):
    x, labels = load_data(train_data, train_labels)
    split_point = int(DATA_SIZE * SPLIT_RATE)
    return (x[:split_point], x[split_point:]), (labels[:split_point], labels[split_point:])


class DataGenerator(object):
    '''
        this class to generate the training data from txt file
        using the gensim to transfer learning
        the model use CBOW and the embedding matrix with a shape 128 for a word
        this model is big enough to build this marking system
        the model is according to the writer in jianshu named __dada__
    '''

    def __init__(self, embedding_model=None, stop_words_path=STOP_WORDS_PATH, data_set_cache_dir=DATA_SET_CACHE_DIR,
                 train_data=TRAIN_DATA, train_labels=TRAIN_LABELS, labels_caregories=LABELS_CAREGORIES,
                 sequence_max_length=SEQUENCE_MAX_LENGTH):
        '''
        :param embedding_model: the gensim model to transfer learning
        :param stop_words_path: stop words path
        :param data_set_cache_dir: temp data cache data
        :param train_data: the data's path to feed the neural network
        :param train_labels: the data's path to feed the neural network
        :param labels_caregories: the caregories of the labels
        :param sequence_max_length: the number of the max length of the model
        if the length is not equals to the length use 0 padding to fill the sequence matrix
        '''
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            raise RuntimeError("the path cant be None")
        self.stop_words_path = stop_words_path
        self.data_set_cache_dir = data_set_cache_dir
        self.train_data = train_data
        self.train_labels = train_labels
        self.labels_caregories = labels_caregories
        self.sequence_max_length = sequence_max_length

    def embedding_words_generator(self):
        '''
        Note:
            using the word2vec (CBOW model) model to make a transfer learning
            to encoding the origin comment to the vector
            to drop out those words which was in stopwords or not in vocabulary
        Others:
            path relational data is in settings.py
            this function is also a generate using lazy-compute
        :return: string list,score
        '''
        # load model
        model = self.embedding_model

        # load stop words
        stop_words_vab = []
        if self.stop_words_path:
            with open(self.stop_words_path, 'r', encoding='utf-8') as f:
                stop_words = f.readlines()
                f.close()
            stop_words_vab = [word[:-1] for word in stop_words]

        for sentence, score in get_words_list():
            # if word not in model vocab or word not in stop_words
            # we change it to a vector use numpy array to store it
            ls = []
            for word in sentence:
                if word in model.vocab and word not in stop_words_vab:
                    ls.append(model[word])
                else:
                    ls.append(np.zeros(shape=(EMBEDDING_UNITS,)))
                    # 0 padding
            ndarr = np.array(ls)
            yield ndarr, score

    def cache(self):
        '''
        Note:
            using embedding_generator to get the words
            and cache them into an .npy file using numpy
        '''
        counter = 0
        score_ls = []
        for word_vector, score in self.embedding_words_generator():
            if not os.path.exists(self.data_set_cache_dir + 'cache-{}.npy'.format(counter)):
                np.save(self.data_set_cache_dir + 'cache-{}.npy'.format(counter), word_vector)
            score_ls.append(score)
            counter += 1
        score_arr = np.array(score_ls, dtype=np.int8)
        if not os.path.exists(self.data_set_cache_dir + 'label.npy'):
            np.save(self.data_set_cache_dir + 'label.npy', score_arr)
        print(counter, 'matrix had been cache')

    def generate(self):
        '''
        Note:
            in this stage u don't need to load the origin model(word2vec)
            this function read the cache and do the two jobs
            one-hot encoding
            sequence_max_length fixed
        Others:
            the details had been written in settings.py
            marks->labels -1->4 -2->3 0->0 1->1 1->2
        '''
        if not (os.path.exists(self.train_data) and os.path.exists(self.train_labels)):
            # one-hot encoding
            labels = np.load(self.data_set_cache_dir + 'label.npy')  # score integer
            # this is encoding mapping
            labels[labels > 0] = 2
            labels[labels == 0] = 1
            labels[labels < 0] = 0
            labels = np.eye(self.labels_caregories)[labels]
            # for 5 labels
            # -1->4 -2->3 0->0 1->1 1->2
            # that you scored 0 1 2 -2 -1
            # for 3 labels only -2 0 2

            # sequences to the unified length
            training_set = []
            for i in range(DATA_SIZE):
                temp = np.load(self.data_set_cache_dir + 'cache-{}.npy'.format(i)).reshape(1, -1, EMBEDDING_UNITS)
                temp = pad_sequences(temp, self.sequence_max_length, dtype='float64', padding='post')
                training_set.append(temp)
            training_set = np.array(training_set).reshape(DATA_SIZE, self.sequence_max_length, EMBEDDING_UNITS)

            np.save(self.train_data, training_set)
            np.save(self.train_labels, labels)


class DataAnalysis(object):
    def __init__(self, data=None):
        self.data = data
        self.data_size = data[0].shape[0]
        self.split_rate = SPLIT_RATE

    def split_data(self):
        x, labels = self.data
        split_point = int(self.data_size * self.split_rate)
        return (x[:split_point], x[split_point:]), (labels[:split_point], labels[split_point:])

    def get_data_size(self):
        return self.data_size

    def get_data_detials(self):
        pass

    def encoder_reduce(self):
        pass

    def visualize(self):
        pass


if __name__ == '__main__':
    '''
    for 5 labels generate
        model = load_embedding_model()
        dg = DataGenerator(model)
        dg.cache()
        dg.generate()
    '''
    # model = load_embedding_model()
    # dg = DataGenerator(model,
    #                    labels_caregories=3,
    #                    train_data=DATA_ROOT + r'iter2/train.npy',
    #                    train_labels=DATA_ROOT + r'iter2/labels.npy',
    #                    )  # for one-hot encoding
    # dg.cache()
    # dg.generate()
    # (x, x_test), (y, y_test) = load_split_data(DATA_ROOT + r'iter2/train.npy', DATA_ROOT + r'iter2/labels.npy')
    # print(y[:100])
    # da = DataAnalysis(load_data())
    # (x, x_test), (y, y_test) = da.split_data()
    # print(x.shape)
    # from keras.models import load_model
    #
    # keras_model = load_model(MODEL_FILE_PATH)
    # print(y[:10])
