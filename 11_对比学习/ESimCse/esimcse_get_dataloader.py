import random

import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
class s_dataset(Dataset):
    def __init__(self,df,tokenizer,dup_rate=0.25,max_len=128,mode='train'):
        self.df = df
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dup_rate = dup_rate
        if mode == 'train':
            self.listdf = self.get_listdf(df)
    def get_listdf(self,df):
        lis = df['text_a'].tolist()
        lis.extend(df['text_b'].tolist())
        return lis
    def __getitem__(self, x):
        if self.mode == 'train':
            text_source = self.listdf[x]
            input_source = self.tokenizer(text_source,truncation = True,max_length = self.max_len)

            text_repeat = self.word_repeat(text_source,self.dup_rate)
            input_repeat = self.tokenizer(text_repeat, truncation=True, max_length=self.max_len)
            return {
                "input_ids_source": torch.as_tensor(input_source['input_ids'], dtype=torch.int),
                "attention_mask_source": torch.as_tensor(input_source['attention_mask'], dtype=torch.int),
                "input_ids_repeat": torch.as_tensor(input_repeat['input_ids'], dtype=torch.int),
                "attention_mask_repeat": torch.as_tensor(input_repeat['attention_mask'], dtype=torch.int),
            }
        else:
            data = self.df.iloc[x]
            text_a,text_b,label = data['text_a'],data['text_b'],data['label']
            input_a = self.tokenizer(text_a, truncation=True, max_length=self.max_len)
            input_b = self.tokenizer(text_b, truncation=True, max_length=self.max_len)
            return {
                "input_ids_a": torch.as_tensor(input_a['input_ids'],dtype=torch.int),
                "attention_mask_a": torch.as_tensor(input_a['attention_mask'], dtype=torch.int),
                "input_ids_b": torch.as_tensor(input_b['input_ids'], dtype=torch.int),
                "attention_mask_b": torch.as_tensor(input_b['attention_mask'], dtype=torch.int),
                "label": label,
            }
    def word_repeat(self,text,dup_rate):
        dup_rate = min(1,dup_rate)
        text_tokens = list(text)
        conduct_num = random.randint(1,max(int(len(text)*dup_rate),1))

        sample =random.sample(range(0,len(text)), conduct_num)
        for num in sample:
            text_tokens[num] += text_tokens[num]
        return "".join(text_tokens)
    def __len__(self):
        if self.mode=='train':
            return len(self.listdf)
        else:
            return len(self.df)
def collate_fn_train(batch):
    mx_len_source = max([len(x['input_ids_source']) for x in batch])
    mx_len_repeat = max([len(x['input_ids_repeat']) for x in batch])

    batch_input_ids_source = torch.zeros((len(batch), mx_len_source), dtype=torch.int)
    batch_attention_mask_source = torch.zeros((len(batch), mx_len_source), dtype=torch.int)
    batch_input_ids_repeat = torch.zeros((len(batch), mx_len_repeat), dtype=torch.int)
    batch_attention_mask_repeat = torch.zeros((len(batch), mx_len_repeat), dtype=torch.int)

    for i, x in enumerate(batch):
        batch_input_ids_source[i, :len(x['input_ids_source'])] = x['input_ids_source']
        batch_attention_mask_source[i, :len(x['attention_mask_source'])] = x['attention_mask_source']
        batch_input_ids_repeat[i, :len(x['input_ids_repeat'])] = x['input_ids_repeat']
        batch_attention_mask_repeat[i, :len(x['attention_mask_repeat'])] = x['attention_mask_repeat']
    return {
        "input_ids_source": batch_input_ids_source,
        "attention_mask_source": batch_attention_mask_source,
        "input_ids_repeat": batch_input_ids_repeat,
        "attention_mask_repeat": batch_attention_mask_repeat,
    }
def collate_fn_dev(batch):
    mx_len_a = max([len(x['input_ids_a']) for x in batch])
    mx_len_b = max([len(x['input_ids_b']) for x in batch])
    batch_input_ids_a = torch.zeros((len(batch),mx_len_a),dtype=torch.int)
    batch_attention_mask_a = torch.zeros((len(batch),mx_len_a),dtype=torch.int)
    batch_input_ids_b = torch.zeros((len(batch), mx_len_b),dtype=torch.int)
    batch_attention_mask_b = torch.zeros((len(batch), mx_len_b),dtype=torch.int)

    label = []
    for i,x in enumerate(batch):
        batch_input_ids_a[i,:len(x['input_ids_a'])] = x['input_ids_a']
        batch_attention_mask_a[i,:len(x['attention_mask_a'])] = x['attention_mask_a']
        batch_input_ids_b[i, :len(x['input_ids_b'])] = x['input_ids_b']
        batch_attention_mask_b[i, :len(x['attention_mask_b'])] = x['attention_mask_b']
        label.append(x['label'])
    return {
        "input_ids_a":batch_input_ids_a,
        "attention_mask_a":batch_attention_mask_a,
        "input_ids_b": batch_input_ids_b,
        "attention_mask_b": batch_attention_mask_b,
        "label":torch.as_tensor(label,dtype=torch.long)
    }
def load_data(batch_size=30):
    train_df = pd.read_csv('../data/ants/train.csv')
    dev_df = pd.read_csv('../data/ants/dev.csv')
    tokenizer = AutoTokenizer.from_pretrained('../../data/chinese-roberta-wwm-ext')
    train_dataset = s_dataset(train_df,tokenizer,mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn_train)
    dev_dataset = s_dataset(dev_df, tokenizer, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_dev)

    return train_dataloader,dev_dataloader

if __name__ == '__main__':
    train_dataloader,dev_dataloader = load_data()
    for data in train_dataloader:
        print(data['input_ids_source'].shape)
        print(data['input_ids_repeat'].shape)
        break
    for data in dev_dataloader:
        print(data['input_ids_a'].shape)
        break