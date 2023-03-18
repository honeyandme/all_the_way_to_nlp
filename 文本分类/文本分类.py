from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import numpy as np

def get_data(file):
    with open(file,mode='r',encoding='utf8') as f:
        all_data = f.read().split('\n')

    all_text, all_label = [],[]
    for d in all_data:
        d = d.split(' ')
        if len(d) !=2:
            continue
        text,lable = d
        try:
            lable = int(lable)
        except Exception as e:
            print(e,f'   标签"{lable}"不是数字')
        else:
            all_text.append(text)
            all_label.append(lable)
    return all_text,all_label
class TextDataset(Dataset):
    def __init__(self,all_text,all_label):
        self.all_text = all_text
        self.all_label = all_label
    def __getitem__(self, index):
        return self.all_text[index],self.all_label[index]
    def deal(self,x):
        x = [word_2_index.get(i, 1) for i in x]
        x = x[:max_len]
        x = x + [0] * (max_len - len(x))
        return x
    def my_collect_fn(self,data):
        global max_len,word_2_index
        batch_text,batch_label = [],[]

        for d in data:
            x,y = d[0],d[1]

            batch_text.append(self.deal(x))
            batch_label.append(y)
        return torch.tensor(batch_text,dtype = torch.float32),torch.tensor(batch_label)
        #return np.array(batch_text),np.array((batch_label))


    def __len__(self):
        return len(all_text)
def build_word2index(all_text):
    mp = {"PAD":0,"UNK":1}
    for sen in all_text:
        for word in sen:
            if word not in mp:
                mp[word] = len(mp)
    return mp
class Model(nn.Module):
    def __init__(self,in_fearure,class_num):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_fearure,10),
            nn.Sigmoid(),
            nn.Linear(10,class_num)
        )
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,label=None):
        x = self.layer(x)
        if label is not None:
            loss = self.loss_fn(x,label)
            return x,loss
        return x
if __name__ == "__main__":
    all_text,all_label = get_data('train.txt')
    assert len(all_text) == len(all_label)
    word_2_index = build_word2index(all_text)
    batch_size = 2
    epoch = 100
    lr = 0.01
    max_len = 7
    train_dataset = TextDataset(all_text,all_label)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.my_collect_fn)

    model = Model(max_len,3)
    opt = torch.optim.Adam(model.parameters(),lr)
    for e in range(epoch):
        loss_sum = 0
        ba_num = 0
        for x,y in train_dataloader:
            opt.zero_grad()
            x,loss = model(x,y)
            loss.backward()
            opt.step()
            loss_sum+=loss
            ba_num +=1
        print(f'当前的loss是{loss_sum/ba_num:.8f}')
    label2ans = ['负向','中性','正向']
    while True:
        x = input('请输入要判别的句子:')
        # x = [word_2_index.get(i,1) for i in x]
        # x = x[:max_len]
        # x = x + [0] * (max_len - len(x))
        x = train_dataset.deal(x)
        x = torch.tensor(x,dtype=torch.float32)
        x = model(x)
        x = torch.argmax(x,dim=-1)
        print(label2ans[x])
