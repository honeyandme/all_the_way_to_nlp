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
    s_path = os.path.join('data','pickle_fen_data_v2.t')
    if(os.path.exists(s_path)):
        with open(s_path,'rb') as f:
            return pickle.load(f)

    text = pd.read_csv(path,encoding='gbk',names=['text'])
    text = text['text'].to_list()
    fen_text = []
    for te in text:
        fen_text.append(jieba.lcut(te))
    with open(s_path, 'wb') as f:
        pickle.dump(fen_text,f)
    return fen_text
def build_word2idx(fen_text):
    mp = {"UNK": 0}
    path = os.path.join('data','pickle_word2idx_v2.t')
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


class Word2Vec(nn.Module):
    def __init__(self, word_size, embedding_num):
        super(Word2Vec, self).__init__()
        self.w1 = nn.Linear(word_size, embedding_num)
        self.w2 = nn.Linear(embedding_num, word_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.w1(x)
        if label is not None:
            x = self.w2(x)
            loss = self.loss_fn(x, label)
            return loss
        return x


class Text_dataset(Dataset):
    def __init__(self, fen_text,grid_num,word2idx):
        self.fen_text = fen_text
        self.grid_num = grid_num
        self.word2idx = word2idx
    def __getitem__(self, idx):
        sentence = fen_text[idx]
        mp = {}
        x, label = [], []

        for i, word in enumerate(sentence):
            all_lin = sentence[i - self.grid_num:i] + sentence[i + 1:i + self.grid_num + 1]
            mp_word =  idx2onehot(self.word2idx.get(word, 0), word2idx_len)
            for lin in all_lin:
                label.append(self.word2idx.get(lin))
                x.append(mp_word)
        return torch.tensor(x, dtype=torch.float32), label

    def my_collect_fn(self, data):
        x, y = None, None
        for d in data:
            if d[0].shape[0] == 0:
                continue
            if x == None:
                x = d[0]
                y = d[1]
                continue
            x = torch.vstack((x, d[0]))
            y+=d[1]
        return x, torch.tensor(y)

    def __len__(self):
        return len(self.fen_text)


if __name__ == '__main__':
    fen_text = get_data('data/数学原始数据.csv')
    word2idx = build_word2idx(fen_text)
    word2idx_len = len(word2idx)
    idx2word = list(word2idx)

    model = torch.load('data/word2idx_2.pt')
    w = model.w1.weight.data.numpy().T

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