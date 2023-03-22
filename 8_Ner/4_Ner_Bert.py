from transformers import BertModel,BertTokenizer
from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
import os
from tqdm import tqdm
from seqeval.metrics import f1_score,precision_score,recall_score
from transformers import logging

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
    word2idx = {"PAD":0,"UNK":100,"O":2}
    for data in all_data:
        for w in data:
            word2idx[w] = word2idx.get(w,len(word2idx))
    return word2idx
class NDataset(Dataset):
    def __init__(self,all_text,all_tag,tokenizer,tag2idx,maxlen=None,is_dev=False):
        self.all_text = all_text
        self.all_tag = all_tag
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.maxlen = maxlen
        self.is_dev=is_dev
    def __getitem__(self, x):
        text,tag = self.all_text[x],self.all_tag[x]
        if self.is_dev:
            max_len = len(text)
        else :
            max_len = self.maxlen
        text_idx = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len + 2,
                              padding="max_length", return_tensors='pt').squeeze(0)
        tag_idx = [0]+[self.tag2idx.get(i,100) for i in tag[:max_len]]+[0]
        tag_idx += [0]*(len(text_idx)-len(tag_idx))

        return text_idx,torch.tensor(tag_idx,dtype=torch.int64)

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
        bert_out_0,bert_out_1 = self.bert(x,return_dict=False)
        pre = self.classifier(bert_out_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre,dim=-1)


if __name__ == "__main__":
    all_text, all_tag = get_data(os.path.join('..', 'data', 'ner', 'BIESO', 'train.txt'))
    dev_text, dev_tag = get_data(os.path.join('..', 'data', 'ner', 'BIESO', 'dev.txt'))

    maxlen = 35
    epoch = 10
    lr = 0.00001
    batch_size = 20

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    tag2idx = build_tag(all_tag)
    idx2tag = list(tag2idx)
    tokenizer = BertTokenizer.from_pretrained(os.path.join('..','data','bert_base_chinese'))

    train_dataset = NDataset(all_text,all_tag,tokenizer,tag2idx,maxlen)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    dev_dataset = NDataset(dev_text, dev_tag, tokenizer, tag2idx, is_dev=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)


    model = Bmodel(len(tag2idx)).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    for e in range(epoch):
        loss_sum = 0
        ba = 0
        model.train()
        for x,y in tqdm(train_dataloader):
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
        now_tag = []
        for x, y in tqdm(dev_dataloader):
            x = x.to(device)
            pre = model(x).reshape(-1)
            pre = [idx2tag[i] for i in pre]
            all_pre.append(pre)
            tagg = [idx2tag[i] for i in y.reshape(-1)]
            now_tag.append(tagg)
        print(f'f1_score={f1_score(all_pre, now_tag):.2f} precision_score={precision_score(all_pre, now_tag):.5f} recall_score={recall_score(all_pre, now_tag):.5f}')
