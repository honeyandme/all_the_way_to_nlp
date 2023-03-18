from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os


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

        for d in data:
            x, y = d[0], d[1]
            x = self.deal(x)
            batch_text.append(x)
            batch_label.append(y)
        batch_text = np.array(batch_text)
        return torch.tensor(batch_text), torch.tensor(batch_label)
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
        self.pos = nn.parameter(pos)

    def forward(self,embedding):
        get = self.pos[:embedding.shape[1],:]
        get = get.unsqueeze(dim=0)
        return embedding+get
class Model(nn.Module):
    def __init__(self, word_size,enbedding_num, class_num):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(word_size,enbedding_num)
        self.hidd = 50
        self.layer1 = nn.Sequential(
            nn.Linear(enbedding_num, self.hidd),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(30 * self.hidd, class_num),
                        # nn.ReLU(),
                        # nn.Linear(128, 64),
                        # nn.Tanh(),
                        # nn.Linear(64, 10)
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.position = Positional(enbedding_num,3000)

    def forward(self, x, label=None):
        x = self.embedding(x)
        x = self.position(x)
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

    max_len = 30
    batch_size = 100
    embedding_len = 150
    lr = 0.001
    epoch = 10
    train_dataset = TextDataset(all_text, all_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=train_dataset.my_collect_fn)

    dev_dataset = TextDataset(dev_text, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False,
                                collate_fn=dev_dataset.my_collect_fn)

    model = Model(len(word_2_index),embedding_len, 10)
    opt = torch.optim.Adam(model.parameters(), lr)

    best_acc = 0
    for e in range(epoch):
        loss_sum = 0
        ba_num = 0
        acc = 0
        for bi, (x, y) in tqdm(enumerate(train_dataloader)):
            opt.zero_grad()
            x, loss = model(x, y)
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
        for bi, (x, y) in tqdm(enumerate(dev_dataloader)):
            x = model(x)
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
