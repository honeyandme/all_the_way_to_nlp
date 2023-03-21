import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import math
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
            dd = len(text_id)
            text_id = text_id + [0] * (self.max_len - len(text_id))
            tag_id = tag_id + [0] * (self.max_len - len(tag_id))
        else:

            text_id = [self.word2idx.get(t, 1) for t in self.all_text[x]]
            tag_id = [self.tag2idx.get(t, 1) for t in self.all_tag[x]]
            dd = len(text_id)

        return torch.tensor(text_id),torch.tensor(tag_id),dd
    def __len__(self):
        return len(self.all_text)

class Positional(nn.Module):
    def __init__(self,d,max_len):
        super().__init__()
        pos = torch.zeros((max_len,d),requires_grad=False)
        t = torch.arange(1,max_len+1,dtype=torch.float32).unsqueeze(1)
        wk = 1.0/10000**(torch.arange(0,d,2)/d)
        angle = wk*t
        pos[:,::2] = np.sin(angle)
        pos[:,1::2] = np.cos(angle)
        self.pos = pos
    def forward(self,embedding):
        get = self.pos[:embedding.shape[1],:]
        get = get.unsqueeze(dim=0).to(embedding.device)
        return embedding+get
class Multi_Head_self_Attention(nn.Module):
    def __init__(self,embedding_num,nhead):
        super().__init__()
        self.nhead = nhead
        self.W_Q = nn.Linear(embedding_num,embedding_num,bias=False)
        self.W_K = nn.Linear(embedding_num, embedding_num,bias=False)
        self.W_V = nn.Linear(embedding_num, embedding_num,bias=False)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        b,l,n = x.shape
        x = x.reshape(b,self.nhead,-1,n)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        score = (Q @ K.transpose(-1,-2))/math.sqrt(x.shape[-1])
        score = self.softmax(score)
        x = score @ V
        x = x.reshape(b,l,n)
        return x
class Norm(nn.Module):
    def __init__(self,embedding_num):
        super().__init__()
        self.l = nn.Linear(embedding_num,embedding_num)
        self.norm = nn.LayerNorm(embedding_num)
    def forward(self,x):
        x = self.l(x)
        x = self.norm(x)
        return x
class Feed_forward(nn.Module):
    def __init__(self,embedding_num,feed_num):
        super().__init__()
        self.l1 = nn.Linear(embedding_num,feed_num)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(feed_num,embedding_num)
    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

class Block(nn.Module):
    def __init__(self,embedding_num,nhead,feed_num):
        super().__init__()
        self.att_lay = Multi_Head_self_Attention(embedding_num, nhead)
        self.norm = Norm(embedding_num)
        self.ffn = Feed_forward(embedding_num, feed_num)
    def forward(self,pos_x):

        att_x = self.att_lay(pos_x)
        norm_x1 = self.norm(att_x)
        norm_x1 = norm_x1 + pos_x

        ffn_x = self.ffn(norm_x1)
        norm_x2 = self.norm(ffn_x)
        norm_x2 = norm_x2 + norm_x1
        return norm_x2
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_num,nhead,feed_num=200,N=2):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.position = Positional(embedding_num,3000)
        self.blocks = nn.Sequential(*[Block(embedding_num,nhead,feed_num)for i in range(N)])

    def forward(self, x,batch_len=None):
        #mask
        if batch_len is not None:
            #mask_martix = torch.ones_like(x)
            mask_martix = torch.ones(size=(*x.shape[:2],1),device=x.device)
            for i in range(len(batch_len)):
                mask_martix[i][batch_len[i]:] = 0
            x = x*mask_martix
        x = self.position(x)
        x = self.blocks(x)
        return x
class NModel(nn.Module):
    def __init__(self,crop_len,embedding_num,class_num):
        super().__init__()
        self.embedding = nn.Embedding(crop_len,embedding_num)

        self.transf = TransformerEncoderLayer(embedding_num,nhead=1)

        self.classifier = nn.Linear(embedding_num,class_num)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,label=None,batch_len=None):
        x = self.embedding(x)
        x = self.transf(x,batch_len)
        pre = self.classifier(x)
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

    train_dataset = NDataset(all_text,all_tag,word2idx,tag2idx,maxlen)
    train_dataloader = DataLoader(train_dataset,batch_size=10,shuffle=True)
    dev_dataset = NDataset(dev_text,dev_tag,word2idx,tag2idx,is_dev=True)
    dev_dataloader = DataLoader(dev_dataset,batch_size=1,shuffle=False)

    model = NModel(crop_len,embedding_num,class_num)
    opt = torch.optim.Adam(model.parameters(),lr = lr)

    for e in range(epoch):
        loss_sum = 0
        ba = 0
        model.train()
        for x,y,batch_len in train_dataloader:
            opt.zero_grad()
            loss = model(x,y,batch_len)
            loss.backward()
            opt.step()
            loss_sum += loss
            ba+=1
        print(f'e = {e} loss={loss_sum/ba:.5f}')

        model.eval()
        all_pre = []
        for x,y,batch_len in dev_dataloader:
            pre = model(x)
            pre = [idx2tag[i] for i in pre]
            all_pre.append(pre)

        print(f'f1_score={f1_score(all_pre,dev_tag):.2f} precision_score={precision_score(all_pre,dev_tag):.5f} recall_score={recall_score(all_pre,dev_tag):.5f}')