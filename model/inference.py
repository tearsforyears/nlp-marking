# coding=utf-8
import tensorflow as tf
from keras.models import Model
from keras.layers import Bidirectional, GRU, Activation, Input, Masking, Dense, Dropout
from keras import regularizers
from settings import *


class KerasModelGenerator(object):
    def __init__(self, sequence_max_length=SEQUENCE_MAX_LENGTH, gru_hidden_units=GRU_HIDDEN_UNITS,
                 gru_dropout=GRU_DROPOUT, dense_hidden_units=DENSE_HIDDEN_UNITS, dense_dropout=DENSE_DROPOUT):
        self.sequence_max_length = sequence_max_length
        self.gru_hidden_units = gru_hidden_units
        self.gru_dropout = gru_dropout
        self.dense_hidden_units = dense_hidden_units
        self.dense_dropout = dense_dropout

    def simple_bi_gru_rnn_inference(self):
        '''
        Note:
            this function build a simple BiDirection-GRU model and DNN with
            GRU (gate recurrent unit) which can encoding the mark of the comment
            the details of the model
            _________________________________________________________________
            Layer (type)                 Output Shape              Param #
            =================================================================
            input_1 (InputLayer)         (None, 30, 128)           0
            _________________________________________________________________
            masking_1 (Masking)          (None, 30, 128)           0
            _________________________________________________________________
            bidirectional_1 (Bidirection (None, 128)               74112
            _________________________________________________________________
            Dense-1 (Dense)              (None, 5)                 645
            _________________________________________________________________
            softmax (Activation)         (None, 5)                 0
            =================================================================
            Total params: 74,757
            Trainable params: 74,757
            Non-trainable params: 0
            _________________________________________________________________
        :return: the keras model
        '''
        # the Bidirectional use two direction to build the network
        input = Input(shape=(self.sequence_max_length, EMBEDDING_UNITS))
        rnn_val = Masking()(input)
        rnn_val = Bidirectional(
            GRU(
                self.gru_hidden_units,
                dropout=self.gru_dropout,
                name='Bidirectional-GRU-network',
            )
        )(rnn_val)

        with tf.name_scope("Connection-Layers"):
            # define dnn to encoding the value of Bidirectional GRU network
            dense_val = Dense(
                self.dense_hidden_units,
                name='Dense-1',
                activity_regularizer=regularizers.l2(REGULAR_NAMDA),
            )(rnn_val)
            dense_val = Dropout(self.dense_dropout)(dense_val)
            output = Activation('softmax', name='softmax')(dense_val)
            model = Model(inputs=input, outputs=output)
            return model

    def bi_gru_stack_rnn_inference(self):
        '''
        Note:
            this function build a stack BiDirection-GRU model and DNN with 3 layers
            GRU (gate recurrent unit) which can encoding the mark of the comment
            the details of the model
            _________________________________________________________________
            Layer (type)                 Output Shape              Param #
            =================================================================
            input_1 (InputLayer)         (None, 30, 128)           0
            _________________________________________________________________
            masking_1 (Masking)          (None, 30, 128)           0
            _________________________________________________________________
            gru_1 (GRU)                  (None, 30, 64)            37056
            _________________________________________________________________
            gru_2 (GRU)                  (None, 30, 64)            24768
            _________________________________________________________________
            bidirectional_1 (Bidirection (None, 128)               49536
            _________________________________________________________________
            Dense-1 (Dense)              (None, 2)                 258
            _________________________________________________________________
            dropout_1 (Dropout)          (None, 2)                 0
            _________________________________________________________________
            softmax (Activation)         (None, 2)                 0
            =================================================================
            Total params: 111,618
            Trainable params: 111,618
            Non-trainable params: 0
            _________________________________________________________________
        :return: the keras model
        '''
        # the Bidirectional use two direction to build the network
        input = Input(shape=(self.sequence_max_length, EMBEDDING_UNITS))
        rnn_val = Masking()(input)

        with tf.name_scope("GRUs-Bidirectional-Neural-Network"):
            rnn_val = GRU(
                self.gru_hidden_units,
                dropout=self.gru_dropout,
                return_sequences=True,
            )(rnn_val)
            rnn_val = GRU(
                self.gru_hidden_units,
                dropout=self.gru_dropout,
                return_sequences=True,
            )(rnn_val)
            rnn_val = Bidirectional(
                GRU(
                    self.gru_hidden_units,
                    dropout=self.gru_dropout,
                )
            )(rnn_val)

        with tf.name_scope("Fully-Connection-Layers"):
            # define dnn to encoding the value of Bidirectional GRU network
            dense_val = Dense(
                3,
                name='Dense-1',
                activity_regularizer=regularizers.l2(REGULAR_NAMDA),
            )(rnn_val)
            dense_val = Dropout(self.dense_dropout)(dense_val)
            output = Activation('softmax', name='softmax')(dense_val)
            model = Model(inputs=input, outputs=output)
            return model


class TensorflowModelGenrator(object):
    pass


if __name__ == '__main__':
    ksg = KerasModelGenerator()
    model = ksg.bi_gru_stack_rnn_inference()
    model.summary()
