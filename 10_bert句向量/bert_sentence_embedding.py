import pandas as pd
import torch
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
class MeanPooling(nn.Module):
    def forward(self,last_hidden_state,mask):
        mask = mask.unsqueeze(dim=2).expand(-1, -1, 768)

        sum_hidden = torch.sum(last_hidden_state*mask,dim=1)
        sum_mask = torch.sum(mask,dim=1)
        sum_mask = torch.clip(sum_mask,min=1e-9)

        mean_hidden = torch.mean(sum_hidden/sum_mask)



        return mean_hidden
class MaxPooling(nn.Module):
    def forward(self, last_hidden_state):
        return torch.max(last_hidden_state, dim=1)
class MinPooling(nn.Module):
    def forward(self, last_hidden_state):
        return torch.min(last_hidden_state, dim=1)
class DeBert_Model(nn.Module):
    def __init__(self,model,get_sentence_embedding_method):
        super(DeBert_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(model)
        self.fc = nn.Linear(self.bert.config.hidden_size,num_class)
        self.loss_fn = nn.CrossEntropyLoss()
        self.mean_pooling = MeanPooling()
        self.get_sentence_embedding_method = get_sentence_embedding_method
        assert get_sentence_embedding_method in ["CLS","MaxPooling","MeanPooling","MinPooling"]
    def forward(self,x,att_mask,y=None):
        last_hidden_state,pooler_output,ft_all_layers= self.bert(x,attention_mask = att_mask,return_dict=False,output_hidden_states=True)#[0][:,-1,:]

        logits=None
        sentence_embedding=None


        if self.get_sentence_embedding_method == "CLS":   #策略1
            logits = self.fc(pooler_output)
            sentence_embedding = pooler_output
        elif self.get_sentence_embedding_method == "MeanPooling":   #策略2
            sentence_embedding = self.mean_pooling(last_hidden_state,att_mask)
            logits = self.fc(pooler_output)
        elif self.get_sentence_embedding_method == "MaxPooling":   #策略3
            pass
        elif self.get_sentence_embedding_method == "MinPooling":   #策略4
            pass
        if y is not None:
            loss = self.loss_fn(logits,y)
            return loss,logits,sentence_embedding
        return logits,sentence_embedding
if __name__ == "__main__":
    train_df = pd.read_csv('data/ants/train.csv')
    dev_df = pd.read_csv('data/ants/dev.csv')

    train_df = train_df[:2000]
    dev_df = dev_df[:500]

    model = '../data/chinese-roberta-wwm-ext'
    num_class = 2
    max_length = 128
    epoch = 10
    batch_size = 30
    lr = 0.00001
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model)

    train_dataset = sen_dataset(train_df)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size,collate_fn=sen_dataset.collate_fn)
    dev_dataset = sen_dataset(dev_df)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size,
                                  collate_fn=sen_dataset.collate_fn)

    model = DeBert_Model(model,"MeanPooling").to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    for e in range(epoch):
        train_model_pre = []
        train_labels = []
        model.train()
        for bi,(x,at_mask,y) in enumerate(tqdm(train_dataloader)):
            x = x.to(device)
            at_mask = at_mask.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss,logits,sentence_embedding = model(x,at_mask,y)
            loss.backward()
            opt.step()

            train_model_pre.extend(torch.argmax(logits,dim=-1).cpu().numpy().tolist())
            train_labels.extend(y.cpu().numpy().tolist())
        train_acc = accuracy_score(train_model_pre,train_labels)
        print(loss)
        model.eval()
        dev_model_pre = []
        dev_labels = []
        for bi,(x,at_mask,y) in enumerate(tqdm(dev_dataloader)):
            x = x.to(device)
            at_mask = at_mask.to(device)
            y = y.to(device)
            logits,sentence_embedding = model(x,at_mask)

            dev_model_pre.extend(torch.argmax(logits,dim=-1).cpu().numpy().tolist())
            dev_labels.extend(y.cpu().numpy().tolist())
        print(dev_model_pre)
        dev_acc = accuracy_score(dev_model_pre,dev_labels)
        print(f'e = {e},train_acc = {train_acc:.3f},dev_acc = {dev_acc:.3f}')

