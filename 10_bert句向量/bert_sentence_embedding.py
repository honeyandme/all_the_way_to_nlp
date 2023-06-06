import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel
class sen_dataset(Dataset):
    def __init__(self,df):
        self.df = df
    def __getitem__(self,x):
        data = self.df.iloc[x]
        text = data['text_a'] + tokenizer.sep_token + data['text_b']
        label = data['label']
        return text,label
        # inputs_a = tokenizer(text,truncation=True,max_length=max_length)
        # return {
        #     "input_ids":torch.tensor(inputs_a['input_ids'],dtype=torch.long),
        #     "attention":torch.tensor(inputs_a['attention_mask'],dtype=torch.long),
        #     "label":torch.tensor(label,dtype=torch.long)
        # }
    def collate_fn(data):
        texts = [x[0] for x in data]
        labels = [x[1] for x in data]
        inputs_a = tokenizer(texts, truncation=True, max_length=max_length,padding='longest')
        return (torch.tensor(inputs_a['input_ids'],dtype=torch.long),
                torch.tensor(inputs_a['attention_mask'], dtype=torch.long),
                torch.tensor(labels,dtype=torch.long) )
    def __len__(self):
        return len(self.df)
class DeBert_Model(nn.Module):
    def __init__(self,model):
        super(DeBert_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.fc = nn.Linear(self.bert.config.hidden_size,num_class)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,att_mask,y=None):
        last_hidden_state,pooler_output,ft_all_layers= self.bert(x,attention_mask = att_mask,return_dict=False,output_hidden_states=True)#[0][:,-1,:]

        logits=None
        sentence_embedding=None

        #策略1
        logits = self.fc(pooler_output)
        sentence_embedding = pooler_output
        if y is not None:
            loss = self.loss_fn(logits,y)
            return loss,logits,sentence_embedding
        return logits,sentence_embedding
if __name__ == "__main__":
    train_df = pd.read_csv('data/ants/train.csv')
    dev_df = pd.read_csv('data/ants/dev.csv')

    model = '../data/chinese-roberta-wwm-ext'
    num_class = 2
    max_length = 128
    epoch = 1
    batch_size = 10
    lr = 1e-5
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model)
    train_dataset = sen_dataset(train_df)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size,collate_fn=sen_dataset.collate_fn)
    model = DeBert_Model(model).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    for e in range(epoch):
        for bi,(x,at_mask,y) in enumerate(tqdm(train_dataloader)):
            x = x.to(device)
            at_mask = at_mask.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss,logits,sentence_embedding = model(x,at_mask,y)
            loss.backward()
            opt.step()
            if bi%100==0:
                print(loss)
