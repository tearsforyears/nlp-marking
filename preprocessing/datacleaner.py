# coding=utf-8
import numpy as np
from settings import *
from preprocessing.embedding import load_data, load_split_data


class Cleaner(object):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def clean(self):
        '''
        Note:
           override this function to clean the data or generate
           the new shape of the data
           the x or the input shape do not change
           but the labels could change in different models
        '''
        clean_x = []
        clean_y = []
        # allow to override
        for i, label in enumerate(self.labels):
            if np.argmax(label) == 3:  # -2
                clean_x.append(self.x[i])
                clean_y.append(np.array([0., 1.]))
            elif np.argmax(label) == 2:  # 2
                clean_x.append(self.x[i])
                clean_y.append(np.array([1., 0.]))
        self.x = np.array(clean_x)
        self.labels = np.array(clean_y)

    def split(self, split_rate=None):
        if split_rate == None:
            split_rate = SPLIT_RATE
        split_point = int(self.x.shape[0] * split_rate)
        return (self.x[:split_point], self.x[split_point:]), (self.labels[:split_point], self.labels[split_point:])

    def shape(self):
        # print(self.x.shape, self.labels.shape)
        return self.x.shape, self.labels.shape

    def get_data(self):
        return self.x, self.labels


def main():
    print(SCORE_LIST2)
    (x, x_test), (y, y_test) = load_split_data()
    print(x.shape, x_test.shape, y.shape, y_test.shape)
    # cleaner = Cleaner(x, labels)
    # cleaner.clean()
    # (x, x_test), (y, y_test) = cleaner.split()
    # print(x.shape, x_test.shape, y.shape, y_test.shape)


if __name__ == '__main__':
    main()
