import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import random
def get_data(path,num = None):
    with open(path,'r') as f:
        all_data = f.read().split('\n')
    if num is not None:
        return all_data[:num]
    return all_data
def build_word2idx(all_data):
    mp = {"UNK":0}
    for sen in all_data:
        for word in sen:
            if word not in mp:
                mp[word] = len(mp)
    return mp
class po_dataset(Dataset):
    def __init__(self,all_data,word2idx):
        self.all_data = all_data
        self.word2idx = word2idx
    def __getitem__(self, index):
        result = [self.word2idx.get(x,0) for x in self.all_data[index]]
        return torch.tensor(result)
    def __len__(self):
        return len(self.all_data)
# class Model(nn.Module):
#     def __init__(self,corpus_len,embedding_num,hidden_size):
#         super(Model, self).__init__()
#         self.embedding = nn.Embedding(corpus_len,embedding_num)
#         self.6_rnn = nn.RNN(input_size=embedding_num,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=False)
#         self.V = nn.Linear(hidden_size,corpus_len)
#         self.loss_fn = nn.CrossEntropyLoss()
#     def forward(self,x,label=None,h_0=None):
#         x = self.embedding(x)
#         if label is not None:
#             o, t = self.6_rnn(x)
#             pre = self.V(o)
#             loss =self.loss_fn(pre.reshape(pre.shape[0]*pre.shape[1],-1),label.reshape(-1))
#             return loss
#         else:
#             o, t = self.6_rnn(x,h_0)
#             pre = self.V(o)
#             return torch.argmax(pre,dim=-1),t
# def auto_generate_poet():
#     global all_data,word2idx,idx2word,model
#     letter = random.choice(idx2word)
#     while letter == '，' or letter == '。':
#         letter = random.choice(idx2word)
#
#     result = [letter]
#     h_0 = torch.zeros((1,1,128))
#     index = torch.tensor([[word2idx[letter]]])
#     for i in range(23):
#         index,h_0 = model(x=index,h_0=h_0)
#         result.append(idx2word[index])
#     return "".join(result)
def auto_generate_poet():
    global all_data,word2idx,idx2word,model,hidden_size,vecmodel
    letter = random.choice(idx2word)
    while letter == '，' or letter == '。':
        letter = random.choice(idx2word)

    result = [letter]
    h_0 = torch.zeros((1,1,128))
    index = word2idx[letter]
    for i in range(23):
        nei = torch.tensor(vecmodel.wv[idx2word[index]]).reshape((1,1,-1))
        index,h_0 = model(x=nei,h_0=h_0)
        result.append(idx2word[index])
    return "".join(result)
class Model(nn.Module):
    def __init__(self,corpus_len,embedding_num,hidden_size):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=embedding_num,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.V = nn.Linear(hidden_size,corpus_len)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,label=None,h_0=None):
        if label is not None:
            o, t = self.rnn(x)
            pre = self.V(o)
            loss =self.loss_fn(pre.reshape(pre.shape[0]*pre.shape[1],-1),label.reshape(-1))
            return loss
        else:
            o, t = self.rnn(x,h_0)
            pre = self.V(o)
            return torch.argmax(pre,dim=-1),t
if __name__=='__main__':
    all_data = get_data(os.path.join('..','data','rnn_data','poetry_5.txt'))

    word2idx = build_word2idx(all_data)
    idx2word = list(word2idx)
    corpus_len = len(word2idx)
    model = torch.load(os.path.join('..','data','rnn_data','best_word2vec.pt'))
    for i in range(10):
        print(auto_generate_poet())




