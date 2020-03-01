# coding=utf-8
import pymongo
from settings import *
import jieba


def mongodb_to_file(path=ORIGIN_COMMENT, collection=COLLECTION_NAME):
    '''
        get the data from mongodb and store them into a test file
    '''
    cli = pymongo.MongoClient('localhost', 27017)
    db = cli[collection]
    dic = {'很差': '-2', '较差': '-1', '还行': '0', '推荐': '1', '力荐': '2'}
    collection_names = db.collection_names()
    with open(path, mode='a', encoding='utf-8') as f:
        for name in collection_names:
            datas = db[name].find()
            f.write('movie_name' + '||' + name + '\n')
            for data in datas:
                if len(data['score']) <= 4 and len(data['comment']) <= 50 and '\n' not in data['comment']:
                    try:
                        f.write(dic[data['score']] + '||' + data['comment'])
                        f.write('\n')
                    except KeyError:
                        print('an key error happened')
    cli.close()


def get_words_list(path=ORIGIN_COMMENT):
    '''
    Note:
        split the sentences to a word list using the api jieba
    Others:
        this is a generate using lazy-compute
        :return: string list
    '''
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line[:line.find('||')] in ('-2', '-1', '0', '1', '2'):
                yield jieba.lcut(line[line.find('||') + 2:-1]), line[:line.find('||')]


if __name__ == '__main__':
    mongodb_to_file()
