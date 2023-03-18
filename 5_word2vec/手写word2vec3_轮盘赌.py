import numpy as np
import pandas as pd
import jieba
import re
from tqdm import tqdm
import pickle
import os
import random
import torch
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

    idx_path = os.path.join('data', 'pickle_word2idx_shouxie.t')
    f_path = os.path.join('data', 'pickle_word2f.t')
    if (os.path.exists(idx_path) and os.path.exists(f_path)):
        with open(idx_path, 'rb') as f:
            word2idx = pickle.load(f)
        with open(f_path, 'rb') as f:
            f = pickle.load(f)
        return word2idx,f
    mp = {"UNK": 0}
    fv = {"UNK": 0}
    all = 0
    for fen in fen_text:
        for word in fen:
            if word not in mp:
                mp[word]= len(mp)
                fv[mp[word]] = 0
            fv[mp[word]]+=1
            all+=1
    with open(idx_path, 'wb') as f:
        pickle.dump(mp,f)
    with open(f_path, 'wb') as f:
        pickle.dump(fv,f)
    return mp,fv

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
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res
if __name__ == '__main__':
    np.random.seed(100)
    fen_text = get_data('data/数学原始数据.csv')

    word2idx,fv = build_word2idx(fen_text)
    fv = sorted(fv.items(), key=lambda x : x[1], reverse=True)

    lunpandu = []
    for word_idx,f in fv:
        for i in range(f):
            lunpandu.append(word_idx)
    l_len = len(lunpandu)
    word2idx_len = len(word2idx)
    idx2word = list(word2idx)
    epoch = 30
    lr = 0.03
    grid_num = 4

    embedding_num = 150
    w1 = np.random.normal(0,0.3,(word2idx_len,embedding_num))
    w2 = np.random.normal(0,0.3, (embedding_num,word2idx_len))
    best_loss = 2e9
    for e in range(epoch):
        loss_sum = 0
        b_num = 0
        for sentence in tqdm(fen_text):
            for i,word in enumerate(sentence):#x:word本身 y:附近词
                other_words = sentence[i-grid_num:i]+sentence[i+1:i+grid_num+1]
                if(len(other_words)==0):
                    continue

                x_idx = np.array([word2idx.get(word, 0)])
                y_idx = [word2idx.get(lin, 0) for lin in other_words]
                fuc_len = 5*len(other_words)
                cha_idx = []
                for i in range(fuc_len):
                    x = random.randint(0, l_len-1)
                    while idx2word[lunpandu[x]] in sentence:
                        x = np.random.randint(0, word2idx_len-1)
                    cha_idx.append(lunpandu[x])
                y_idx = np.array(y_idx+cha_idx)
                h = w1[x_idx]
                pre = h @ w2[:, y_idx]

                soft_pre = sigmoid(pre)
                label = np.array([[1]*len(other_words)+[0]*fuc_len])
                loss = -np.mean(label*np.log(soft_pre)+(1-label)*np.log(1-soft_pre))

                G = soft_pre-label
                delta_w2 = h.T @ G
                delta_h = G @ w2[:,y_idx].T
                delta_w1 = delta_h#x.T @ delta_h

                w1[x_idx,:] -= lr * delta_w1
                w2[:,y_idx] -= lr * delta_w2

                loss_sum += loss
                b_num += 1
        loss_sum/=b_num
        if loss_sum <best_loss:
            best_loss = loss_sum
            np.save('./data/w1_lr5.npy', w1)
            print(f'e={e}  loss={loss_sum:.5f}------------------------------------>best')
        else:
            print(f'e={e}  loss={loss_sum:.5f}')

