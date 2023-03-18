import gensim
import os
import pickle
import pandas as pd
import jieba
import re
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

def get_stop_words(path):
    with open(path, 'r', encoding='utf8') as f:
        return f.read().split('\n')
def get_data(path):
    s_path = os.path.join('data', 'pickle_fen_data_shouxie.t')
    if(os.path.exists(s_path)):
        with open(s_path,'rb') as f:
            return pickle.load(f)
    pat = '[\u4e00-\u9fa5]'
    stop_words = get_stop_words(os.path.join('data', 'stopwords.txt'))
    text = pd.read_csv(path,encoding='gbk',names=['text'])
    text = text['text'].to_list()
    fen_text = []
    for te in text:
        te = re.findall(pat,te)
        te = "".join(te)
        tc = jieba.lcut(te)
        tc = [x for x in tc if x not in stop_words]
        fen_text.append(tc)
    with open(s_path, 'wb') as f:
        pickle.dump(fen_text,f)
    return fen_text
if __name__ =='__main__':
    fen_text = get_data('data/数学原始数据.csv')
    model = gensim.models.Word2Vec(fen_text,vector_size=150,batch_words=20)

    print()