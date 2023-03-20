import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import os
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
    word2idx = {"PAD":0,"UNK":1,"O":2}
    for data in all_data:
        for w in data:
            word2idx[w] = word2idx.get(w,len(word2idx))
    return word2idx
class NDataset(Dataset):
    def __init__(self,all_text,all_tag,word2idx,tag2idx,max_len=30):
        self.all_text = all_text
        self.all_tag = all_tag
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len
    def __getitem__(self, x):
        text_id = [self.word2idx.get(t,1) for t in self.all_text[x][:self.max_len]]
        tag_id = [self.tag2idx.get(t,1) for t in self.all_tag[x][:self.max_len]]
        text_id = text_id +[0]*(self.max_len-len(text_id))
        tag_id = tag_id + [0]*(self.max_len-len(tag_id))
        # 高版本自动转tensor
        return text_id,tag_id
    def __len__(self):
        return len(self.all_text)
if __name__ == '__main__':
    all_text,all_tag = get_data(os.path.join('..','data','ner','BIESO','train.txt'))
    maxlen = 30
    word2idx = build_word(all_text)
    tag2idx = build_word(all_tag)
    dataset = NDataset(all_text,all_tag,word2idx,tag2idx,maxlen)
    dataloader = DataLoader(dataset,batch_size=10,shuffle=False)

    for x,y in dataloader:
        print()
