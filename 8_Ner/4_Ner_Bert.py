from transformers import BertModel,BertTokenizer
from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
import os
from tqdm import tqdm
from seqeval.metrics import f1_score,precision_score,recall_score
from transformers import logging
import pickle
logging.set_verbosity_warning()

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
def build_tag(all_data):
    if(os.path.exists('tag2idx.pickle')):
        with open('tag2idx.pickle','rb') as f:
            return pickle.load(f)
    word2idx = {"PAD":0,"UNK":1,"O":2}
    for data in all_data:
        for w in data:
            word2idx[w] = word2idx.get(w,len(word2idx))
    with open('tag2idx.pickle', 'wb') as f:
        pickle.dump(word2idx,f)
    return word2idx
class NDataset(Dataset):
    def __init__(self,all_text,all_tag,tokenizer,tag2idx):
        self.all_text = all_text
        self.all_tag = all_tag
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
    def __getitem__(self, x):
        text,tag = self.all_text[x],self.all_tag[x]
        return text,tag,len(text)
    def collate_fn(self,data):
        batch_len = []
        for d in data:
            batch_len.append(d[2])
        batch_max = max(batch_len)
        batch_text, batch_tag = [],[]
        for d in data:
            text,tag = d[:2]
            text_idx = self.tokenizer.encode(text, add_special_tokens=True, max_length=batch_max + 2,
                                             padding="max_length")
            tag_idx = [0] + [self.tag2idx.get(i, 100) for i in tag] + [0]
            tag_idx += [0] * (len(text_idx) - len(tag_idx))
            batch_text.append(text_idx)
            batch_tag.append(tag_idx)
        return torch.tensor(batch_text),torch.tensor(batch_tag,dtype=torch.int64),torch.tensor(batch_len,dtype=torch.int64)
    def __len__(self):
        return len(self.all_text)

class Bmodel(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.bert = BertModel.from_pretrained(os.path.join('..','data','bert_base_chinese'))
        # for name,param in self.bert.named_parameters():
        #     # print(name,param.shape)
        #     if '0' in name or 'pooler' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        self.classifier = nn.Linear(768,class_num)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,label=None):
        bert_out_0,bert_out_1 = self.bert(x,attention_mask=(x>0),return_dict=False)
        pre = self.classifier(bert_out_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)


if __name__ == "__main__":
    mode = 'BIESO'#指定标注模式
    all_text, all_tag = get_data(os.path.join('..', 'data', 'ner', mode, 'train.txt'))
    dev_text, dev_tag = get_data(os.path.join('..', 'data', 'ner', mode, 'dev.txt'))

    epoch = 10
    lr = 0.00001
    batch_size = 20
    full_train = False #是否加载已经训练的结果

    #apple gpu
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tag2idx = build_tag(all_tag)
    idx2tag = list(tag2idx)
    tokenizer = BertTokenizer.from_pretrained(os.path.join('..','data','bert_base_chinese'))

    train_dataset = NDataset(all_text,all_tag,tokenizer,tag2idx)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,collate_fn=train_dataset.collate_fn)

    dev_dataset = NDataset(dev_text, dev_tag, tokenizer, tag2idx)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False,collate_fn=train_dataset.collate_fn)


    model = Bmodel(len(tag2idx)).to(device)
    if os.path.exists(f'best_{mode}.pt'):
        model.load_state_dict(torch.load(f'best_{mode}.pt'))
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    best_f1 = -1
    print(f'train on {device}...')
    for e in range(epoch):
        loss_sum = 0
        ba = 0
        model.train()
        for x,y,batch_len in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = model(x,y)
            loss.backward()
            opt.step()
            loss_sum += loss
            ba += 1
        print(f'e = {e} loss={loss_sum / ba:.5f}')

        model.eval()
        all_pre = []
        for x, y,batch_len in tqdm(dev_dataloader):
            x = x.to(device)
            pre = model(x).reshape(-1)
            pre = [idx2tag[i] for i in pre[1:batch_len+1]]
            all_pre.append(pre)
        f1 = f1_score(all_pre, dev_tag)
        if f1>best_f1:
            best_f1 = f1
            print(f'f1_score={f1:.2f}---------------------------------->best')
            torch.save(model.state_dict(), f'best_{mode}.pt')
        else:
            print(f'f1_score={f1:.2f}')