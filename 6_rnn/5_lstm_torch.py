import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
from tqdm import tqdm
import pickle
def get_data(file):
    with open(file, mode='r', encoding='utf8') as f:
        all_data = f.read().split('\n')
        all_data = sorted(all_data,key = lambda x:len(x))
        all_data = all_data

    all_text, all_label = [], []
    for d in all_data:
        d = d.split('	')
        if len(d) != 2:
            continue
        text, lable = d
        try:
            lable = int(lable)
        except Exception as e:
            print(e, f'   标签"{lable}"不是数字')
        else:
            all_text.append(text)
            all_label.append(lable)
    return all_text, all_label
def build_word2index(all_text):
    path = os.path.join('..','data','rnn_data','word2idx.pkl')
    if os.path.exists(path):
        with open(path,'rb') as f:
            return pickle.load(f)
    mp = {"PAD": 0, "UNK": 1}
    for sen in all_text:
        for word in sen:
            if word not in mp:
                mp[word] = len(mp)
    with open(path,'wb') as f:
        pickle.dump(mp,f)
    return mp

class TextDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        x,y = self.all_text[index], self.all_label[index]
        x = [word2idx.get(i, 1) for i in x]
        return x,y

    def my_collect_fn(self, data):
        batch_text, batch_label = [], []

        batch_max = 0
        for d in data:
            batch_max = max(batch_max,len(d[0]))
        for d in data:
            x, y = d[0], d[1]
            x = x + [0] * (batch_max - len(x))
            batch_text.append(x)
            batch_label.append(y)
        batch_text = np.array(batch_text)
        return torch.tensor(batch_text), torch.tensor(batch_label)
        # return np.array(batch_text),np.array((batch_label))

    def __len__(self):
        return len(self.all_text)
class RNN_Model(nn.Module):
    def __init__(self,embedding_num,hidden_fea):
        super(RNN_Model, self).__init__()
        self.hidden_fea = hidden_fea
        self.W = nn.Linear(embedding_num,hidden_fea)
        self.U = nn.Linear(hidden_fea,hidden_fea)
        self.tanh = nn.Tanh()
    def forward(self,x):
        t = torch.zeros((x.shape[0],self.hidden_fea),device=x.device)
        O = torch.zeros((x.shape[0],x.shape[1],self.hidden_fea),device=x.device)
        for i in range(x.shape[1]):
            x_ = x[:,i]
            x_ = self.W(x_)
            x_ = x_+t
            x_ = self.tanh(x_)

            t  = self.U(x_)
            O[:,i] = t
            #O.append(o)

        return t,O

class LSTM_model(nn.Module):
    def __init__(self,embedding_num,hidden_num):
        super(LSTM_model, self).__init__()
        self.F = nn.Linear(embedding_num+hidden_num,hidden_num)
        self.I = nn.Linear(embedding_num + hidden_num, hidden_num)
        self.C = nn.Linear(embedding_num + hidden_num, hidden_num)
        self.O = nn.Linear(embedding_num + hidden_num, hidden_num)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self,x_emb,c_0=None,h_0=None):#embedding
        if c_0==None:
            c_0 = torch.zeros((x_emb.shape[0],hidden_num),device=x_emb.device)
        if h_0==None:
            h_0 = torch.zeros((x_emb.shape[0],hidden_num),device=x_emb.device)
        all_h = torch.zeros((*x_emb.shape[:2],hidden_num),device=x_emb.device)
        for i in range(x_emb.shape[1]):
            x = x_emb[:,i]
            x = torch.cat((x,c_0),dim=1)
            f_ = self.F(x)
            i_ = self.I(x)
            c_ = self.C(x)
            o_ = self.O(x)

            f__ = self.sigmoid(f_)
            i__ = self.sigmoid(i_)
            c__ = self.tanh(c_)
            o__ = self.sigmoid(o_)

            c_now = f__* c_0 + i__* c__
            ct = self.tanh(c_now)
            h_now = ct*o__

            c_0 = c_now
            h_0 = h_now

            all_h[:,i] = h_0
        return all_h,(c_0,h_0)
class Model(nn.Module):
    def __init__(self,word_size,embedding_num,hidden_fea,class_num):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(word_size,embedding_num)
        self.rnn = LSTM_model(embedding_num,hidden_fea)
        self.classifier = nn.Linear(hidden_fea,class_num)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x,y=None):
        x_emb = self.embedding(x)
        all_h,(c,h) = self.rnn(x_emb)
        pre = self.classifier(c)
        if y is not None:
            loss = self.loss_fn(pre,y)
            return pre,loss
        return pre


if __name__ =='__main__':
    x = np
    all_text, all_label = get_data(os.path.join('..','data', 'train.txt'))
    dev_text, dev_label = get_data(os.path.join('..','data', 'dev.txt'))
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    embedding_len = 128
    hidden_num = 100
    batch_size = 30
    epoch = 10
    lr = 0.001
    max_len = 30

    word2idx = build_word2index(all_text)
    train_dataset = TextDataset(all_text,all_label)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,collate_fn=train_dataset.my_collect_fn,shuffle = False)
    dev_dataset = TextDataset(dev_text, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False,
                                collate_fn=dev_dataset.my_collect_fn)


    model = Model(len(word2idx), embedding_len, hidden_num,10)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr)


    best_acc = 0
    for e in range(epoch):
        model.train()
        loss_sum = 0
        ba_num = 0
        acc = 0
        for bi, (x, y) in tqdm(enumerate(train_dataloader)):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            x, loss = model(x, y)
            loss.backward()
            opt.step()

            loss_sum += loss
            ba_num += 1
            batch_acc = torch.sum(torch.argmax(x, dim=-1) == y)
            acc += batch_acc
            if bi % 1000 == 0:
                print(f'loss={loss:.5f} acc = {batch_acc / x.shape[0]:.8f}')
        print(f'e={e + 1}当前的loss是{loss_sum / ba_num:.8f} acc={acc / len(train_dataset) :.8f}')
        model.eval()
        acc = 0
        for bi, (x, y) in tqdm(enumerate(dev_dataloader)):
            x = x.to(device)
            y = y.to(device)
            x = model(x)
            acc += torch.sum(torch.argmax(x, dim=-1) == y)
        acc = acc / len(dev_dataset)
        if acc > best_acc:
            best_acc = acc
            print(f'验证集准确率为 {acc :.8f}--------------------------------->best')
        else:
            print(f'验证集准确率为 {acc :.8f}')
