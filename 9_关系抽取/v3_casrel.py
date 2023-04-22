import os
import random
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizerFast
def read_data(path):
    with open(path,'r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    ev_data = []
    for data in all_data:
        if len(data)>0:
            ev_data.append(eval(data))
    return ev_data
def build_rel_2_index():
    with open(os.path.join('data2','rel.txt'),'r',encoding='utf-8') as f:
        index_2_rel = f.read().split('\n')
    rel_2_index = {k:v for v,k in enumerate(index_2_rel)}
    return rel_2_index,index_2_rel
class C_Dataset(Dataset):
    def __init__(self,all_data,rel_2_index,tokenizer):
        self.all_data = all_data
        self.rel_2_index = rel_2_index
        self.tokenizer = tokenizer
    def __getitem__(self, x):
        info = self.all_data[x]
        token_text = self.tokenizer(info['text'],add_special_tokens=True,return_offsets_mapping=True)
        info['input_ids']=token_text['input_ids']
        info['offset_mapping']=token_text['offset_mapping']
        return self.parase_info(info)
    def parase_info(self,info):
        dic = {
            'text' : info['text'],
            'input_ids': info['input_ids'],
            'offset_mapping': info['offset_mapping'],
            'sub_head_ids':[],
            'sub_tail_ids':[],
            'triple_list':[],
            'triple_id_list': []
        }
        for spo in info['spo_list']:
            subject = spo['subject']
            predicate = spo['predicate']
            object = spo['object']['@value']
            dic['triple_list'].append((subject,predicate,object))

            subject_ids = self.tokenizer.encode(subject,add_special_tokens=False)
            sub_pos = self.find_pos(info['input_ids'],subject_ids)
            if sub_pos is None:
                continue
            sub_head, sub_tail = sub_pos

            object_ids = self.tokenizer.encode(object, add_special_tokens=False)
            obj_pos = self.find_pos(info['input_ids'], object_ids)
            if obj_pos is None:
                continue
            obj_head, obj_tail = obj_pos

            dic['sub_head_ids'].append(sub_head)
            dic['sub_tail_ids'].append(sub_tail)

            dic['triple_id_list'].append((
                [sub_head,sub_tail],
                [self.rel_2_index[predicate]],
                [obj_head,obj_tail],
            ))

        return dic
    def find_pos(self,input_ids,ids):
        l = len(ids)
        for i in range(len(input_ids)):
            if input_ids[i:i+l]==ids:
                return i,i+l-1
        return None
    def multihot(self,hot_len,pos):
        return [1 if i in pos else 0 for i in range(hot_len)]
    def operate_data(self,batch_data):
        batch_text = {
            "text":[],
            "input_ids": [],
            "offset_mapping": [],
            "triple_list": [],
        }
        batch_mask = []
        batch_sub = {
            "heads_seq":[],
            "tails_seq":[]
        }
        batch_sub_rnd = {
            "heads_seq": [],
            "tails_seq": []
        }
        batch_obj_rel = {
            "heads_mx":[],
            "tails_mx":[],
        }

        max_len = 0
        for item in batch_data:
            max_len = max(max_len,len(item['input_ids']))

        for item in batch_data:
            input_ids = item['input_ids']
            input_len = len(input_ids)
            pad_len = max_len-input_len
            input_ids += [0]* pad_len

            batch_text['text'].append(item['text'])
            batch_text['input_ids'].append(input_ids)
            batch_text['offset_mapping'].append(item['offset_mapping'])
            batch_text['triple_list'].append(item['triple_list'])

            mask = [1]*input_len+[0]*pad_len

            batch_mask.append(mask)

            sub_heads_seq = self.multihot(max_len,item['sub_head_ids'])
            sub_tails_seq = self.multihot(max_len,item['sub_tail_ids'])

            batch_sub['heads_seq'].append(sub_heads_seq)
            batch_sub['tails_seq'].append(sub_tails_seq)

            #随机挑选
            sub_rnd_head,sub_end_tail = random.choice(item['triple_id_list'])[0]
            #sub_rnd_head_2_tail = self.multihot(max_len,[sub_rnd_head,sub_end_tail])
            sub_rnd_head_seq = self.multihot(max_len,[sub_rnd_head])
            sub_rnd_tail_seq = self.multihot(max_len, [sub_end_tail])

            batch_sub_rnd['heads_seq'].append(sub_rnd_head_seq)
            batch_sub_rnd['tails_seq'].append(sub_rnd_tail_seq)

            obj_head_mx = [[0]*len(self.rel_2_index) for i in range(max_len)]
            obj_tail_mx = [[0]*len(self.rel_2_index) for i in range(max_len)]
            for triple in item['triple_id_list']:
                rel_id = triple[1][0]
                head_id,tail_id = triple[2]
                obj_head_mx[head_id][rel_id] = 1
                obj_tail_mx[tail_id][rel_id] = 1

            batch_obj_rel['heads_mx'].append(obj_head_mx)
            batch_obj_rel['tails_mx'].append(obj_tail_mx)
        return batch_text,batch_mask,batch_sub,batch_sub_rnd,batch_obj_rel


    def __len__(self):
        return len(self.all_data)

class Cas_Model(nn.Module):
    def __init__(self,model_name,rel_num):
        super(Cas_Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        self.sub_head_linear = nn.Linear(768,1)
        self.sub_tail_linear = nn.Linear(768,1)

        self.obj_head_linear = nn.Linear(768,rel_num)
        self.obj_tail_linear = nn.Linear(768,rel_num)

        self.sigmoid = nn.Sigmoid()
    def get_text_encode(self,input_ids,mask):
        bert_0,bert_1 = self.bert(input_ids,attention_mask=mask,return_dict=False)
        return bert_0
    def get_sub_pre(self,text_encode):
        sub_head_pre = self.sigmoid(self.sub_head_linear(text_encode))
        sub_tail_pre = self.sigmoid(self.sub_tail_linear(text_encode))

        return sub_head_pre,sub_tail_pre
    def get_obj_pre(self,text_encode,heads_seq,tails_seq):
        heads_seq = heads_seq.unsqueeze(1).float()
        tails_seq = tails_seq.unsqueeze(1).float()

        W1 = heads_seq @ text_encode
        W2 = tails_seq @ text_encode
        text_encode = text_encode +(W1+W2)/2
        obj_head_pre = self.sigmoid(self.obj_head_linear(text_encode))
        obj_tail_pre = self.sigmoid(self.obj_tail_linear(text_encode))
        return obj_head_pre,obj_tail_pre
    def forward(self,input,mask):
        input_ids,heads_seq,tails_seq = input
        text_encode = self.get_text_encode(input_ids,mask)

        sub_head_pre, sub_tail_pre = self.get_sub_pre(text_encode)
        obj_head_pre, obj_tail_pre = self.get_obj_pre(text_encode,heads_seq,tails_seq)

        return sub_head_pre, sub_tail_pre, obj_head_pre, obj_tail_pre

if __name__ == "__main__":
    train_data = read_data(os.path.join('data2','duie_dev.json'))
    rel_2_index,index_2_rel = build_rel_2_index()

    batch_size = 10
    epoch = 1
    model_name = '../data/bert_base_chinese'

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_dataset = C_Dataset(train_data,rel_2_index,tokenizer)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size,collate_fn=train_dataset.operate_data)

    model = Cas_Model(model_name,len(rel_2_index))
    for e in range(epoch):
        for batch_text,batch_mask,batch_sub,batch_sub_rnd,batch_obj_rel in train_dataloader:

            input = (
                torch.tensor(batch_text['input_ids']),
                torch.tensor(batch_sub_rnd['heads_seq']),
                torch.tensor(batch_sub_rnd['tails_seq']),

            )
            mask = torch.tensor(batch_mask)
            model.forward(input,mask)