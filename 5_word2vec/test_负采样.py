import numpy as np
import pandas as pd
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import os
#from sklearn.metrics.pairwise import cosine_similarity
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


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



if __name__ == '__main__':
    fen_text = get_data('data/数学原始数据.csv')
    word2idx = build_word2idx(fen_text)
    word2idx_len = len(word2idx)
    idx2word = list(word2idx)

    w = np.load('data/w1.npy')

    while (True):
        word = input('请输入词:')
        if word in word2idx:
            word_vec = w[word2idx[word]]
            lis = np.array([np.abs(cosine_similarity(word_vec,w[idx])) for idx in range(word2idx_len)])
            arg_lis = lis.argsort().tolist()[::-1][:10]
            score_lis = [lis[i] for i in arg_lis]
            arg_lis = [idx2word[i] for i in arg_lis]

            for x,y in zip(arg_lis,score_lis):
               print(f'{x}:{y}')



    # while(True):
    #     x1,x2 = input('请输入两个词').split(' ')
    #     if x1 in word2idx and x2 in word2idx:
    #         x1_vec = w[word2idx[x1]]
    #
    #         x2_vec = w[word2idx[x2]]
    #         #sim = cosine_similarity(x1_vec.reshape(-1,1),x2_vec.reshape(-1,1))
    #         sim = cosine_similarity(x1_vec,x2_vec)
    #         # print(f'{x1}和 {x2}的距离是{sim:.3f}')
    #         print(sim)