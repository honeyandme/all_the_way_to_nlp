import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
class s_dataset(Dataset):
    def __init__(self,df,tokenizer,max_len=128,mode='train'):
        self.df = df
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        if mode == 'train':
            self.listdf = self.get_listdf(df)
    def get_listdf(self,df):
        lis = df['text_a'].tolist()
        lis.extend(df['text_b'].tolist())
        return lis
    def __getitem__(self, x):
        if self.mode == 'train':
            text = self.listdf[x]
            input_a = self.tokenizer(text,truncation = True,max_length = self.max_len)
            return torch.as_tensor(input_a["input_ids"],dtype=torch.int), \
                   torch.as_tensor(input_a["attention_mask"], dtype=torch.int)
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

    def __len__(self):
        if self.mode=='train':
            return len(self.listdf)
        else:
            return len(self.df)
def collate_fn_train(batch):
    mx_len = max([len(x[0]) for x in batch])
    batch_input_ids = torch.zeros((len(batch)*2,mx_len),dtype=torch.int)
    batch_attention_mask = torch.zeros((len(batch)*2,mx_len),dtype=torch.int)

    for i,x in enumerate(batch):
        batch_input_ids[2*i,:len(x[0])] = x[0]
        batch_input_ids[2 * i+1, :len(x[0])] = x[0]
        batch_attention_mask[2*i,:len(x[1])] = x[1]
        batch_attention_mask[2 * i+1, :len(x[1])] = x[1]
    return {
        "input_ids":batch_input_ids,
        "attention_mask":batch_attention_mask
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
        print(data['input_ids'].shape)
        break
    for data in dev_dataloader:
        print(data['input_ids_a'].shape)
        break