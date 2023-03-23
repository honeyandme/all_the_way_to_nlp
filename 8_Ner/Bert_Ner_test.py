from transformers import BertModel,BertTokenizer
from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
import os
import pickle
from tqdm import tqdm
from seqeval.metrics import f1_score,precision_score,recall_score
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

if __name__=='__main__':
    mode = 'BIESO'  # 指定标注模式
    with open('tag2idx.pickle', 'rb') as f:
        tag2idx=pickle.load(f)
    idx2tag = list(tag2idx)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained(os.path.join('..', 'data', 'bert_base_chinese'))

    model = Bmodel(len(tag2idx)).to(device)
    model.load_state_dict(torch.load(f'best_{mode}.pt'))

    while(True):
        text = input('请输入句子:')
        text_idx = tokenizer.encode(text, add_special_tokens=True,return_tensors='pt')
        text_idx = text_idx.to(device)
        pre = model.forward(text_idx)[0][1:-1].tolist()
        pre = [idx2tag[x] for x in pre]
        for x,y in zip(text,pre):
            print(x,y)