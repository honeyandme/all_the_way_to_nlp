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
class Model(nn.Module):
    def __init__(self,corpus_len,embedding_num,hidden_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(corpus_len,embedding_num)
        self.rnn = nn.RNN(input_size=embedding_num,hidden_size=hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.V = nn.Linear(hidden_size,corpus_len)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,label=None,h_0=None):
        x = self.embedding(x)
        if label is not None:
            o, t = self.rnn(x)
            pre = self.V(o)
            loss =self.loss_fn(pre.reshape(pre.shape[0]*pre.shape[1],-1),label.reshape(-1))
            return loss
        else:
            o, t = self.rnn(x,h_0)
            pre = self.V(o)
            return torch.argmax(pre,dim=-1),t
def auto_generate_poet():
    global all_data,word2idx,idx2word,model,hidden_size
    letter = random.choice(idx2word)
    while letter == '，' or letter == '。':
        letter = random.choice(idx2word)

    result = [letter]
    h_0 = torch.zeros((1,1,hidden_size))
    index = torch.tensor([[word2idx[letter]]])
    for i in range(23):
        index,h_0 = model(x=index,h_0=h_0)
        result.append(idx2word[index])
    return "".join(result)


if __name__=='__main__':
    all_data = get_data(os.path.join('..','data','rnn_data','poetry_5.txt'))

    word2idx = build_word2idx(all_data)
    idx2word = list(word2idx)
    corpus_len = len(word2idx)

    epoch = 1000
    batch_size = 50
    lr = 2e-3
    embedding_num = 100
    hidden_size = 128

    train_dataset = po_dataset(all_data,word2idx)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    model = Model(corpus_len,embedding_num,hidden_size)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    best_loss = 2e9
    for e in range(epoch):
        loss_sum = 0
        ba_num = 0
        for x in train_dataloader:
            opt.zero_grad()
            x_input = x[:,:-1]
            x_label = x[:,1:]
            loss = model(x=x_input,label=x_label)
            loss.backward()
            opt.step()

            loss_sum+=loss
            ba_num += 1
        loss_sum /= ba_num

        if loss_sum<best_loss:
            best_loss = loss_sum
            print(f'e={e} loss={loss_sum:.5f}----------------------->best')
            torch.save(model,os.path.join('..','data','rnn_data','best_duan__.pt'))
        else:
            print(f'e={e} loss={loss_sum:.5f}')
        print(auto_generate_poet())





