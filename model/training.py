# coding=utf-8
import tensorflow as tf
from settings import *
from model.inference import KerasModelGenerator
from preprocessing.embedding import load_data, load_split_data
from preprocessing.datacleaner import Cleaner
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import os


class Training(object):
    def __init__(self, keras_model=None, optimizer=None, tensorboard=None, checkpoint=None, loss=None,
                 batch_size=BATCH_SIZE, epoch=EPOCH, model_file_path=MODEL_FILE_PATH, last_model_name=LAST_MODEL_NAME,
                 lr=LEARNING_RATE, decay_rate=DECAY_RATE):
        self.keras_model = keras_model
        self.optimizer = optimizer
        self.tensorboard = tensorboard
        self.checkpoint = checkpoint
        self.loss = loss
        self.batch_size = batch_size
        self.epoch = epoch
        self.model_file_path = model_file_path
        self.last_model_name = last_model_name
        # initialize
        if self.optimizer is None:
            self.optimizer = Adam(lr, decay=decay_rate)
        if self.tensorboard is None:
            self.tensorboard = TensorBoard(log_dir=LOG_DIR)
        if self.checkpoint is None:
            self.checkpoint = ModelCheckpoint(self.model_file_path, save_best_only=True)
        if self.loss is None:
            self.loss = 'categorical_crossentropy'

    def training_by_keras(self, x, y, x_test, y_test):
        '''
        Note:
            use the Adam as the optimizer
            use the cross_entropy as the loss function
            use breakpoint to split the learning stage
            use checkpoint to save data
        '''
        if self.keras_model is None:
            raise RuntimeError("[error:] the model is None")

        if os.path.exists(self.model_file_path):
            self.keras_model.load_weights(self.model_file_path)

        # compile the model and train the model
        self.keras_model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy']
        )
        hist = self.keras_model.fit(
            x, y,
            callbacks=[self.tensorboard, self.checkpoint],
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(x_test, y_test)
        )
        # save the last time data
        self.keras_model.save(self.last_model_name)
        return hist

    def training_by_tensorflow(self, x, y, x_test, y_test):
        pass


def main():
    '''test function'''
    # load model
    ksg = KerasModelGenerator()
    model = ksg.bi_gru_stack_rnn_inference()
    # load data
    x, labels = load_data()
    cleaner = Cleaner(x, labels)
    cleaner.clean()
    (x, x_test), (y, y_test) = cleaner.split()
    print(x.shape, x_test.shape, y.shape, y_test.shape)
    train_op = Training(
        model,
        model_file_path=DATA_ROOT + r'iter2/marking-model-v2.h5',
        last_model_name=DATA_ROOT + r'iter2/marking-model-v2-final.h5'
    )
    train_op.training_by_keras(x, y, x_test, y_test)


def main2():
    '''test function'''
    # load model
    ksg = KerasModelGenerator()
    model = ksg.bi_gru_stack_rnn_inference()
    # load data
    (x, x_test), (y, y_test) = load_split_data(
        train_data=DATA_ROOT + 'iter3/train.npy',
        train_labels=DATA_ROOT + 'iter3/labels.npy',
    )
    print(x.shape, x_test.shape, y.shape, y_test.shape)
    train_op = Training(
        model,
        model_file_path=DATA_ROOT + r'iter3/marking-model-v2.h5',
        last_model_name=DATA_ROOT + r'iter3/marking-model-v2-final.h5'
    )
    train_op.training_by_keras(x, y, x_test, y_test)


if __name__ == '__main__':
    main2()
