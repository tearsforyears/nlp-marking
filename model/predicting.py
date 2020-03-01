# coding=utf-8
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from utils.clock import runtime
from preprocessing.embedding import load_embedding_model
from settings import *
from keras.models import load_model
import jieba


class Predictor(object):
    '''
        this class provide a command-line interface of using keras model
        and provide the online predicting interface
    '''

    def __init__(self, keras_model, gensim_model, score_list=SCORE_LIST):
        self.keras_model = keras_model
        self.gensim_model = gensim_model
        self.score_list = score_list

    def __println(self, str):
        blocks = int((98 - len(str)) / 2)
        print("-", ' ' * blocks, str)

    def Interactive_Command_Line(self):
        '''
            this function create a user interface to let the user the control the model
            to predict user class sentences
        '''
        print('—' * 100)
        self.__println('this a nlp marking system')
        self.__println('use rnn to predict the score')
        self.__println('build in 2018')
        self.__println('author:tearsforyears(守护全世界最好的赖美云)')
        print('—' * 100)
        while 1:
            command = input(">>>")
            if command == 'exit':
                break
            elif command == 'predict':
                input_sent = input("input your sentence:")
                arr = self.get_sent_matrix(input_sent)
                prob = self.predict(arr)
                print(self.score_list, '\n', prob)
                print('the max prop is', self.score_list[np.argmax(prob)])
            elif command == 'help':
                print('options:predict,exit,help')
            else:
                print('wrong command,use help to get more')

    @runtime("processing embedding time")
    def get_sent_matrix(self, sent):
        '''
            this is the online predicting core api
            this function make a chinese sentence to a matrix using 0 padding
            only for predicting this could be a component of a pipeline used
            by other programs
        '''
        model = self.gensim_model
        ls = []
        for word in jieba.lcut(sent):
            if word in model.vocab:
                ls.append(model[word])
            else:
                ls.append(np.zeros(shape=(EMBEDDING_UNITS,)))
                # 0 padding
        ndarr = np.array(ls).reshape(1, -1, EMBEDDING_UNITS)
        ndarr = pad_sequences(ndarr, SEQUENCE_MAX_LENGTH, dtype='float64', padding='post')
        return ndarr

    @runtime("predict time")
    def predict(self, x):
        '''
            this use keras model to predict and also provide others program
            to use this interface to be a component or theirs pipeline
        '''
        return self.keras_model.predict(x)


def main():
    '''
        the test function to test command line interface and
        the online predicting api
    '''
    print('<---loading models--->')
    keras_model = load_model(DATA_ROOT + r'iter3/marking-model-v3-final.h5')
    gensim_model = load_embedding_model()
    predictor = Predictor(keras_model, gensim_model, score_list=SCORE_LIST3)
    predictor.Interactive_Command_Line()
    # that you scored 0 1 2 -2 -1


if __name__ == '__main__':
    main()
