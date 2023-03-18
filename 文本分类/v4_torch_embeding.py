from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random

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
    def __init__(self, all_text, all_label,max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.max_len = max_len
    def __getitem__(self, index):
        x,y = self.all_text[index], self.all_label[index]
        x = x[:self.max_len]
        x = [word_2_index.get(i, 1) for i in x]
        x = x + [0] * (self.max_len - len(x))
        return torch.tensor(x),torch.tensor(y)
    # def my_collect_fn(self, data):
    #     global max_len
    #     batch_text, batch_label = [], []
    #
    #     for d in data:
    #         x, y = d[0], d[1]
    #         x = x[:max_len]
    #         x = x + [0] * (max_len - len(x))
    #         batch_text.append(x)
    #         batch_label.append(y)
    #     batch_text = np.array(batch_text)
    #     return torch.tensor(batch_text), torch.tensor(batch_label)
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


class Model(nn.Module):
    def __init__(self, num_embedding,embedding_dim, class_num):
        super(Model, self).__init__()
        self.hidd = 50
        self.embedding = nn.Embedding(num_embedding,embedding_dim)
        self.embedding.weight.requires_grad = True
        self.layer1 = nn.Sequential(
            nn.Linear(embedding_dim, self.hidd),
            nn.Tanh(),
            # nn.Dropout(0.4)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(30 * self.hidd, 128),
            nn.Tanh(),
            # nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.Tanh(),
            # nn.Dropout(0.2),
            nn.Linear(64, class_num)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        x = self.embedding(x)
        x = self.layer1(x)
        x = x.view(-1, 30 * self.hidd)
        x = self.layer2(x)
        if label is not None:
            loss = self.loss_fn(x, label)
            return x, loss
        return x


def same_seed(myseed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
same_seed(6666)
import os

if __name__ == "__main__":
    all_text, all_label = get_data(os.path.join('data', 'train.txt'))
    dev_text, dev_label = get_data(os.path.join('data', 'dev.txt'))
    assert len(all_text) == len(all_label)
    assert len(dev_text) == len(dev_text)
    word_2_index = build_word2index(all_text)

    batch_size = 100
    epoch = 100
    lr = 0.001
    mx_len = 30
    train_dataset = TextDataset(all_text, all_label,mx_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  )

    dev_dataset = TextDataset(dev_text, dev_label,mx_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False,
                                )

    model = Model(len(word_2_index) ,embedding_dim=150, class_num=10)
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
