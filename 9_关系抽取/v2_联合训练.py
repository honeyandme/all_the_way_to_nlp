import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizer
from seqeval.metrics import f1_score,precision_score,recall_score
def get_data(path):
    with open(path,'r',encoding='utf8') as f:
        all_data = f.read().split('\n')
    all_text,all_tag,all_relation = [],[],[]
    for data in all_data:
        data = eval(data)
        ent1,pos1 = data['h']['name'],data['h']['pos']
        ent2,pos2 = data['t']['name'],data['t']['pos']



        text = data['text']

        tag = ['O']*len(text)
        tag[pos1[0]:pos1[1]] = ['B'] +['I']*(len(ent1)-1)
        tag[pos2[0]:pos2[1]] = ['B'] + ['I'] * (len(ent2) - 1)
        text = list(text)
        all_text.append(text)
        all_tag.append(tag)
        all_relation.append(data['relation'])
    return all_text,all_tag,all_relation

def build_relation_2_index(relation):
    mp = {}
    for rel in relation:
        mp[rel] = mp.get(rel,len(mp))
    return mp

class Ndataset(Dataset):
    def __init__(self,all_text,all_tag,tokenizer,tag_2_index,max_len,relation,relation_2_index):
        self.all_text = all_text
        self.all_tag = all_tag
        self.tokenizer = tokenizer
        self.tag_2_index = tag_2_index
        self.max_len = max_len
        self.relation = relation
        self.relation_2_index = relation_2_index


    def __getitem__(self, x):
        text,tag = self.all_text[x][:500],self.all_tag[x][:self.max_len]
        rel = self.relation[x]

        text_idx = self.tokenizer.encode(text,add_special_tokens=True,max_length=self.max_len+2,padding="max_length",truncation=True)
        tag_idx = [0]+[self.tag_2_index[i] for i in tag]+[0]
        tag_idx += [0] * (len(text_idx) - len(tag_idx))
        assert len(text_idx)==len(tag_idx)
        return torch.tensor(text_idx),torch.tensor(tag_idx),len(tag),self.relation_2_index[rel]
    def __len__(self):
        return len(self.all_text)

class Bmodel(nn.Module):
    def __init__(self,model_name,tag_num,rel_num):
        super(Bmodel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.RNN = nn.RNN(768,384,batch_first=True,bidirectional=True)
        self.tag_classifier = nn.Linear(768,tag_num)
        self.rel_classifier = nn.Linear(768, rel_num)
        self.loss_fn = nn.CrossEntropyLoss()

        self.alpha = 0.6
    def forward(self,x,y=None,rel=None):
        bert_out0,bert_out1 = self.bert(x,return_dict=False)
        lstm_out,_ = self.RNN(bert_out0)
        tag_pre = self.tag_classifier(lstm_out)

        rel_pre = self.rel_classifier(bert_out1)
        if y is not None:
            loss1 = self.loss_fn(tag_pre.reshape(-1,tag_pre.shape[-1]),y.reshape(-1))
            loss2 = self.loss_fn(rel_pre,rel)
            loss = loss1*self.alpha+loss2*(1-self.alpha)
            return loss
        return torch.argmax(tag_pre,dim=-1),torch.argmax(rel_pre,dim=-1)
if __name__ == "__main__":
    train_text,train_tag,train_relation = get_data(os.path.join('data','train_small.jsonl'))
    dev_text, dev_tag,dev_relation = get_data(os.path.join('data', 'val_small.jsonl'))
    tag_2_index = {'[PAD]':0,'O':1,'B':2,'I':3}
    index_2_tag = list(tag_2_index)

    relation_2_index = build_relation_2_index(train_relation+dev_relation)

    batch_size = 40
    model_name = '../data/bert_base_chinese'
    epoch = 10
    max_len = 40
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    lr = 1e-5

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = Ndataset(train_text,train_tag,tokenizer,tag_2_index,max_len,train_relation,relation_2_index)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

    dev_dataset = Ndataset(dev_text, dev_tag, tokenizer, tag_2_index, max_len,dev_relation,relation_2_index)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size)

    model = Bmodel(model_name,len(tag_2_index),len(relation_2_index)).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    for e in range(epoch):
        loss_sum = 0
        for x,y,batch_len,batch_rel in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            batch_rel = batch_rel.to(device)
            opt.zero_grad()
            loss = model(x,y,batch_rel)
            loss.backward()
            opt.step()
            loss_sum+=loss*x.shape[0]
        print(f"e={e} loss={loss_sum/len(train_dataset):.5f}")
        all_pre,all_tag = [],[]
        acc = 0
        for x,y,batch_len,batch_rel in tqdm(dev_dataloader):
            x = x.to(device)
            pre,rel_pre = model(x)
            acc+=torch.sum(rel_pre.cpu()==batch_rel)
            for p,q,l in zip(pre,y,batch_len):
                p = p.cpu().tolist()
                q = q.cpu().tolist()
                p = p[1:l+1]
                q = q[1:l+1]
                p = [index_2_tag[i] for i in p]
                q = [index_2_tag[i] for i in q]
                all_pre.append(p)
                all_tag.append(q)
        f1 = f1_score(all_pre,all_tag)
        print(f"f1_score={f1:.5f} acc={acc/len(dev_dataset)}")
