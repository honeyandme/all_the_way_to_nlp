import numpy as np
import pandas as pd
import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import os

def get_data(path):
    s_path = os.path.join('data','pickle_fen_data_v2.t')
    if(os.path.exists(s_path)):
        with open(s_path,'rb') as f:
            return pickle.load(f)
    stop_words = get_stop_words(os.path.join('data','stopwords.txt'))
    text = pd.read_csv(path,encoding='gbk',names=['text'])
    text = text['text'].to_list()
    fen_text = []
    for te in text:
        tc = jieba.lcut(te)
        tc = [x for x in tc if x not in stop_words]
        fen_text.append(tc)
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

    epoch = 100
    lr = 1e-3
    grid_num = 1

    model = Word2Vec(word2idx_len, 150)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = Text_dataset(fen_text,grid_num,word2idx)
    best_loss = 10000
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=dataset.my_collect_fn)
    for e in range(epoch):
        loss_sum = 0
        b_num = 0
        for bi, (x, label) in enumerate(tqdm(dataloader), 1):
            opt.zero_grad()
            loss = model(x, label)

            loss.backward()
            opt.step()

            loss_sum += loss
            b_num += 1
        loss_sum /= b_num
        if loss_sum < best_loss:
            best_loss = loss_sum
            print(f'e={e} ----------loss={loss_sum:.8f}-------------------------------->best')
            torch.save(model, 'data/word2idx.pt')
        else:
            print(f'e={e} ----------loss={loss_sum:.8f}')
