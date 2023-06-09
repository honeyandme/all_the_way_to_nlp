import pandas as pd
import torch
import numpy as np
import random
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel
class sen_dataset(Dataset):
    def __init__(self,df):
        self.df = df
    def __getitem__(self,x):
        data = self.df.iloc[x]
        return data['text_a'],data['text_b'],data['label']
    def collate_fn(data):
        inputa = [x[0] for x in data]
        inputb = [x[1] for x in data]
        labels = [x[2] for x in data]
        inputs_a = tokenizer(inputa, truncation=True, max_length=max_length,padding='longest')
        inputs_b = tokenizer(inputb, truncation=True, max_length=max_length, padding='longest')
        return {
                "input_ids_a":torch.tensor(inputs_a['input_ids'],dtype=torch.long),
                "att_a":torch.tensor(inputs_a['attention_mask'], dtype=torch.long),
                "input_ids_b": torch.tensor(inputs_b['input_ids'], dtype=torch.long),
                "att_b":torch.tensor(inputs_b['attention_mask'], dtype=torch.long),
                "label":torch.tensor(labels,dtype=torch.long)
                }
    def __len__(self):
        return len(self.df)
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self,last_hidden_state,mask):
        mask = mask.unsqueeze(dim=2).expand(last_hidden_state.shape).float()
        sum_hidden = torch.sum(last_hidden_state*mask,dim=1)
        sum_mask = torch.sum(mask,dim=1)
        sum_mask = torch.clamp(sum_mask,min=1e-9)
        mean_hidden = sum_hidden/sum_mask



        return mean_hidden
class Bert_Model(nn.Module):
    def __init__(self):
        super(Bert_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.meanpooling = MeanPooling()
        self.linear = nn.Linear(768*3,2)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,input_ids_a,att_a,input_ids_b,att_b,label=None):
        last_hidden_state_a, pooler_output_a = self.bert(input_ids_a,att_a,return_dict = False)
        last_hidden_state_b, pooler_output_b = self.bert(input_ids_b, att_b, return_dict=False)

        u = self.meanpooling(last_hidden_state_a,att_a)
        v = self.meanpooling(last_hidden_state_b,att_b)

        u_v = torch.abs(torch.sub(u,v))

        s_emb = torch.cat([u,v,u_v],dim = -1)

        logits = self.linear(s_emb)

        if label is not None:
            loss = self.loss_fn(logits,label)
            return loss,logits
        return logits

if __name__ == "__main__":
    train_df = pd.read_csv('data/ants/train.csv')[:2000]
    dev_df = pd.read_csv('data/ants/dev.csv')[:500]

    model = '../data/chinese-roberta-wwm-ext'
    num_class = 2
    max_length = 128
    epoch = 10
    batch_size = 10
    lr = 0.00001
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)

    train_dataset = sen_dataset(train_df)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=sen_dataset.collate_fn)
    dev_dataset = sen_dataset(dev_df)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size,
                                collate_fn=sen_dataset.collate_fn)

    model = Bert_Model().to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)

    for e in range(epoch):
        train_model_pre = []
        train_labels = []
        model.train()
        for bi,data in enumerate(tqdm(train_dataloader)):
            input_ids_a = data['input_ids_a'].to(device)
            att_a = data['att_a'].to(device)
            input_ids_b = data['input_ids_b'].to(device)
            att_b = data['att_b'].to(device)
            label = data['label'].to(device)

            loss,logits = model(input_ids_a,att_a,input_ids_b,att_b,label)
            loss.backward()
            if bi%5==0:
                opt.step()
                opt.zero_grad()
            train_model_pre.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            train_labels.extend(label.cpu().numpy())
        train_acc = accuracy_score(train_model_pre, train_labels)

        model.eval()
        dev_model_pre = []
        dev_labels = []
        for bi,data in enumerate(tqdm(dev_dataloader)):
            input_ids_a = data['input_ids_a'].to(device)
            att_a = data['att_a'].to(device)
            input_ids_b = data['input_ids_b'].to(device)
            att_b = data['att_b'].to(device)
            label = data['label'].to(device)

            logits = model(input_ids_a,att_a,input_ids_b,att_b)
            dev_model_pre.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            dev_labels.extend(label.cpu().numpy())
        dev_acc = accuracy_score(dev_model_pre, dev_labels)
        print(f'e = {e},train_acc = {train_acc:.3f},dev_acc = {dev_acc:.3f}')