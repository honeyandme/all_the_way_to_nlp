import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import os
from seqeval.metrics import f1_score,precision_score,recall_score
def get_data(path):
    all_text,all_tag = [],[]
    with open(path,'r',encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen,tag = [],[]
    for data in all_data:
        data = data.split(' ')
        if(len(data)!=2):
            if len(sen)>2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te,ta = data
        sen.append(te)
        tag.append(ta)
    return all_text,all_tag
def build_word(all_data):
    word2idx = {"PAD":0,"UNK":1}
    for data in all_data:
        for w in data:
            word2idx[w] = word2idx.get(w,len(word2idx))
    return word2idx
def build_tag(all_data):
    word2idx = {"PAD":0,"UNK":1,"O":2}
    for data in all_data:
        for w in data:
            word2idx[w] = word2idx.get(w,len(word2idx))
    return word2idx
class NDataset(Dataset):
    def __init__(self,all_text,all_tag,word2idx,tag2idx,max_len=50,is_dev=False):
        self.all_text = all_text
        self.all_tag = all_tag
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len
        self.is_dev=is_dev
    def __getitem__(self, x):
        if self.is_dev==False:
            text_id = [self.word2idx.get(t,1) for t in self.all_text[x][:self.max_len]]
            tag_id = [self.tag2idx.get(t,1) for t in self.all_tag[x][:self.max_len]]
            text_id = text_id + [0] * (self.max_len - len(text_id))
            tag_id = tag_id + [0] * (self.max_len - len(tag_id))
        else:
            text_id = [self.word2idx.get(t, 1) for t in self.all_text[x]]
            tag_id = [self.tag2idx.get(t, 1) for t in self.all_tag[x]]

        return torch.tensor(text_id),torch.tensor(tag_id)
    def __len__(self):
        return len(self.all_text)

class NModel(nn.Module):
    def __init__(self,crop_len,embedding_num,hidden_num,class_num):
        super().__init__()
        self.embedding = nn.Embedding(crop_len,embedding_num)

        #self.rnn = nn.LSTM(input_size = embedding_num,hidden_size=hidden_num,num_layers=2,batch_first=True,bidirectional=True)
        self.rnn = nn.GRU(input_size = embedding_num,hidden_size=hidden_num,num_layers=1,batch_first=True,bidirectional=True)

        self.classifier = nn.Linear(hidden_num*2,class_num)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,label=None):
        x = self.embedding(x)
        o,t = self.rnn(x)
        pre = self.classifier(o)# [10, 50, 150]
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1,pre.shape[2]),y.reshape(-1))
            return loss
        return torch.argmax(pre,dim=-1).reshape(-1)

if __name__ == '__main__':
    all_text,all_tag = get_data(os.path.join('..','data','ner','BIESO','train.txt'))
    dev_text, dev_tag = get_data(os.path.join('..', 'data', 'ner', 'BIESO', 'dev.txt'))
    maxlen = 50
    word2idx = build_word(all_text)
    tag2idx = build_tag(all_tag)
    idx2tag = list(tag2idx)

    crop_len = len(word2idx)
    embedding_num = 150
    class_num = len(tag2idx)
    lr = 0.001
    epoch = 20
    batch_size = 10
    hidden_num = 150

    train_dataset = NDataset(all_text,all_tag,word2idx,tag2idx,maxlen)
    train_dataloader = DataLoader(train_dataset,batch_size=10,shuffle=True)
    dev_dataset = NDataset(dev_text,dev_tag,word2idx,tag2idx,is_dev=True)
    dev_dataloader = DataLoader(dev_dataset,batch_size=1,shuffle=False)

    model = NModel(crop_len,embedding_num,hidden_num,class_num)
    opt = torch.optim.Adam(model.parameters(),lr = lr)

    for e in range(epoch):
        loss_sum = 0
        ba = 0
        model.train()
        for x,y in train_dataloader:
            opt.zero_grad()
            loss = model(x,y)
            loss.backward()
            opt.step()
            loss_sum += loss
            ba+=1
        print(f'e = {e} loss={loss_sum/ba:.5f}')

        model.eval()
        all_pre = []
        for x,y in dev_dataloader:
            pre = model(x)
            pre = [idx2tag[i] for i in pre]
            all_pre.append(pre)
        print(f'f1_score={f1_score(all_pre,dev_tag):.2f} precision_score={precision_score(all_pre,dev_tag):.5f} recall_score={recall_score(all_pre,dev_tag):.5f}')