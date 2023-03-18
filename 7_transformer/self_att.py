from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import math
def get_data(file):
    with open(file, mode='r', encoding='utf8') as f:
        all_data = f.read().split('\n')

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


class TextDataset(Dataset):
    def __init__(self, all_text, all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        x,y = self.all_text[index], self.all_label[index]
        x = [word_2_index.get(i, 1) for i in x]
        return x,y

    def deal(self, x):
        x = x[:max_len]
        x = x + [0] * (max_len - len(x))
        return x

    def idx2onehot(self, x, len):
        res = [0] * len
        res[x] = 1
        return res

    def my_collect_fn(self, data):
        global max_len, word_2_index
        batch_text, batch_label = [], []

        batch_len = []
        for d in data:
            x, y = d[0], d[1]
            batch_len.append(len(x))
            x = self.deal(x)

            batch_text.append(x)
            batch_label.append(y)
        batch_text = np.array(batch_text)
        return torch.tensor(batch_text), torch.tensor(batch_label),torch.tensor(batch_len)
        # return np.array(batch_text),np.array((batch_label))

    def __len__(self):
        return len(self.all_text)


def build_word2index(all_text):
    mp = {"PAD": 0, "UNK": 1}
    for sen in all_text:
        for word in sen:
            if word not in mp:
                mp[word] = len(mp)
    return mp
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
    def __init__(self, word_size,embedding_num, class_num,nhead,feed_num,N):
        super().__init__()
        self.embedding = nn.Embedding(word_size,embedding_num)
        self.hidd = 50
        self.layer1 = nn.Sequential(
            nn.Linear(embedding_num, self.hidd),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(30 * self.hidd, class_num),

        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.position = Positional(embedding_num,3000)
        self.blocks = nn.Sequential(*[Block(embedding_num,nhead,feed_num)for i in range(N)])

    def forward(self, x,label=None,batch_len=None):
        x = self.embedding(x)
        #mask
        if batch_len is not None:
            #mask_martix = torch.ones_like(x)
            mask_martix = torch.ones(size=(*x.shape[:2],1),device=x.device)
            for i in range(len(batch_len)):
                mask_martix[i][batch_len[i]:] = 0
            x = x*mask_martix
        x = self.position(x)
        x = self.blocks(x)
        x = self.layer1(x)
        x = x.view(-1, 30 * self.hidd)
        x = self.layer2(x)
        if label is not None:
            loss = self.loss_fn(x, label)
            return x, loss
        return x



if __name__ == "__main__":
    all_text, all_label = get_data(os.path.join('..','文本分类','data', 'train.txt'))
    dev_text, dev_label = get_data(os.path.join('..','文本分类','data', 'dev.txt'))
    assert len(all_text) == len(all_label)
    assert len(dev_text) == len(dev_text)
    word_2_index = build_word2index(all_text)
    # device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    max_len = 30
    batch_size = 200
    embedding_len = 150
    lr = 0.001
    epoch = 10
    nhead = 3
    N = 2
    feed_num = embedding_len#int(embedding_len*1.2)
    train_dataset = TextDataset(all_text, all_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=train_dataset.my_collect_fn)

    dev_dataset = TextDataset(dev_text, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False,
                                collate_fn=dev_dataset.my_collect_fn)

    model = TransformerEncoderLayer(len(word_2_index),embedding_len, 10,nhead,feed_num,N)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr)

    best_acc = 0
    for e in range(epoch):
        loss_sum = 0
        ba_num = 0
        acc = 0
        for bi, (x, y,batch_len) in tqdm(enumerate(train_dataloader)):
            opt.zero_grad()
            x = x.to(device)
            y = y.to(device)
            batch_len = batch_len.to(device)
            x, loss = model(x, y,batch_len)
            loss.backward()
            opt.step()

            loss_sum += loss
            ba_num += 1
            batch_acc = torch.sum(torch.argmax(x, dim=-1) == y)
            acc += batch_acc
            if bi % 400 == 0:
                print(f'loss={loss:.5f} acc = {batch_acc / x.shape[0]:.8f}')
        print(f'e={e + 1}当前的loss是{loss_sum / ba_num:.8f} acc={acc / len(train_dataset) :.8f}')
        acc = 0
        for bi, (x, y,batch_len) in tqdm(enumerate(dev_dataloader)):
            x = x.to(device)
            y = y.to(device)
            batch_len = batch_len.to(device)
            x = model(x,batch_len=batch_len)

            acc += torch.sum(torch.argmax(x, dim=-1) == y)
        acc = acc/len(dev_dataset)
        if acc> best_acc:
            best_acc = acc
            print(f'验证集准确率为 {acc :.8f}--------------------------------->best')
        else:
            print(f'验证集准确率为 {acc :.8f}')

    label2ans = ['负向', '中性', '正向']
    # while True:
    #     x = input('请输入要判别的句子:')
    #     # x = [word_2_index.get(i,1) for i in x]
    #     # x = x[:max_len]
    #     # x = x + [0] * (max_len - len(x))
    #     x = train_dataset.deal(x)
    #     x = torch.tensor(x,dtype=torch.float32)
    #     x = model(x)
    #     x = torch.argmax(x,dim=-1)
    #     print(label2ans[x])
