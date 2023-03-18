import numpy as np
import pandas as pd
import jieba
import re
from tqdm import tqdm
import pickle
import os

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
def build_word2idx(fen_text):
    mp = {"UNK": 0}
    path = os.path.join('data', 'pickle_word2idx_shouxie.t')
    if (os.path.exists(path)):
        with open(path, 'rb') as f:
            return pickle.load(f)
    for fen in fen_text:
        for word in fen:
            if word not in mp:
                mp[word]= len(mp)
    with open(path, 'wb') as f:
        pickle.dump(mp,f)
    return mp

def get_stop_words(path):
    with open(path, 'r', encoding='utf8') as f:
        return f.read().split('\n')



def idx2onehot(x, len):
    res = [0] * len
    res[x] = 1
    return res

def soft_max(x):
    x = np.exp(x)
    e_sum = np.sum(x,axis=1,keepdims=True)
    return x/e_sum
if __name__ == '__main__':
    fen_text = get_data('data/数学原始数据.csv')

    word2idx = build_word2idx(fen_text)
    word2idx_len = len(word2idx)

    epoch = 100
    lr = 1e-3
    grid_num = 2

    embedding_num = 100
    w1 = np.random.normal(0,1,(word2idx_len,embedding_num))
    w2 = np.random.normal(0,1, (embedding_num,word2idx_len))
    for e in range(epoch):
        for sentence in tqdm(fen_text):
            for i,word in enumerate(sentence):
                other_words = sentence[i-grid_num:i]+sentence[i+1:i+grid_num+1]

                x = np.array([idx2onehot(word2idx.get(word, 0), word2idx_len)]* len(other_words))
                y = np.array([idx2onehot(word2idx.get(x,0),word2idx_len) for x in other_words])

                h = x @ w1
                pre = h@ w2
                soft_pre = soft_max(pre)

                loss = -np.sum(y*np.log(soft_pre))

                G = soft_pre-y
                delta_w2 = h.T @ G
                delta_h = G @ w2.T
                delta_w1 = x.T @ delta_h

                w1 -= lr * delta_w1
                w2 -= lr * delta_w2

