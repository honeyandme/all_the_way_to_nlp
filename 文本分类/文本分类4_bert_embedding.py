from transformers import BertTokenizer,BertModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


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


class Model(nn.Module):
    def __init__(self, word_size,enbedding_num=768, class_num=10):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(word_size, enbedding_num)
        self.embedding.weight.data = torch.tensor(np.load(os.path.join('data','bert_word_embedding.npy')))
        self.embedding.weight.requires_grad = False
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

    def forward(self, x, label=None):
        x = self.embedding(x)
        x = self.layer1(x)
        x = x.view(-1, 30 * self.hidd)
        x = self.layer2(x)
        if label is not None:
            loss = self.loss_fn(x, label)
            return x, loss
        return x


import os

if __name__ == "__main__":
    all_text, all_label = get_data(os.path.join('data', 'train.txt'))
    dev_text, dev_label = get_data(os.path.join('data', 'dev.txt'))
    assert len(all_text) == len(all_label)
    assert len(dev_text) == len(dev_text)
    #word_2_index = build_word2index(all_text)
    with open(os.path.join('data','bert_base_chinese','vocab.txt'),'r') as f:
        index2word = f.read().split('\n')
    word_2_index = {w:i for i,w in enumerate(index2word)}
    batch_size = 100
    epoch = 10
    lr = 0.001
    max_len = 30
    train_dataset = TextDataset(all_text , all_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=train_dataset.my_collect_fn)

    dev_dataset = TextDataset(dev_text, dev_label)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False,
                                collate_fn=dev_dataset.my_collect_fn)

    #bert_tokenizer = BertTokenizer.from_pretrained(os.path.join('data','bert_base_chinese'))
    model = Model(len(word_2_index))
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


