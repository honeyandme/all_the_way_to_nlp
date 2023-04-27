import os
import random
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizerFast
def read_data(path,num=None):
    with open(path,'r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    ev_data = []
    for data in all_data:
        if len(data)>0:
            ev_data.append(eval(data))
    if num is not None:
        return ev_data[:num]
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

            sub_heads_seq = multihot(max_len,item['sub_head_ids'])
            sub_tails_seq = multihot(max_len,item['sub_tail_ids'])

            batch_sub['heads_seq'].append(sub_heads_seq)
            batch_sub['tails_seq'].append(sub_tails_seq)

            #随机挑选
            sub_rnd_head,sub_end_tail = random.choice(item['triple_id_list'])[0]
            #sub_rnd_head_2_tail = self.multihot(max_len,[sub_rnd_head,sub_end_tail])
            sub_rnd_head_seq = multihot(max_len,[sub_rnd_head])
            sub_rnd_tail_seq = multihot(max_len, [sub_end_tail])

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
def multihot(hot_len,pos):
    return [1 if i in pos else 0 for i in range(hot_len)]
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
        sub_head_pre = torch.sigmoid(self.sub_head_linear(text_encode))
        sub_tail_pre = torch.sigmoid(self.sub_tail_linear(text_encode))

        return sub_head_pre,sub_tail_pre
    def get_obj_pre(self,text_encode,heads_seq,tails_seq):
        heads_seq = heads_seq.unsqueeze(1).float()
        tails_seq = tails_seq.unsqueeze(1).float()

        W1 = torch.matmul(heads_seq , text_encode)
        W2 = torch.matmul(tails_seq , text_encode)
        text_encode = text_encode +(W1+W2)/2
        obj_head_pre = torch.sigmoid(self.obj_head_linear(text_encode))
        obj_tail_pre = torch.sigmoid(self.obj_tail_linear(text_encode))
        return obj_head_pre,obj_tail_pre
    def forward(self,input,mask):
        if len(input)==3:#train mode
            input_ids,heads_seq,tails_seq = input
            text_encode = self.get_text_encode(input_ids,mask)

            sub_head_pre, sub_tail_pre = self.get_sub_pre(text_encode)
            obj_head_pre, obj_tail_pre = self.get_obj_pre(text_encode,heads_seq,tails_seq)

            return sub_head_pre, sub_tail_pre, obj_head_pre, obj_tail_pre
        else:#dev mode
            input_ids = input[0]
            text_encode = self.get_text_encode(input_ids, mask)
            sub_head_pre, sub_tail_pre = self.get_sub_pre(text_encode)
            return text_encode,(sub_head_pre, sub_tail_pre)
    def cal_loss(self,x,y,mask):
        x = x.squeeze(-1)
        if x.shape!=mask.shape:
            mask = mask.unsqueeze(-1)
        x=x*mask
        return nn.functional.binary_cross_entropy(x,y.float())

    def loss_fn(self,pre_y,true_y,mask):
        sub_head_pre, sub_tail_pre, obj_head_pre, obj_tail_pre = pre_y
        sub_head_label, sub_tail_label, obj_head_label, obj_tail_label = true_y

        loss1 = self.cal_loss(sub_head_pre,sub_head_label,mask)
        loss2 = self.cal_loss(sub_tail_pre, sub_tail_label, mask)
        loss3 = self.cal_loss(obj_head_pre, obj_head_label, mask)
        loss4 = self.cal_loss(obj_tail_pre, obj_tail_label, mask)
        loss = loss1+loss2+loss3+loss4
        return loss

def get_eneity(text,hp,tp,offset_map,mask):
    all_entity = []
    i,j = 0,0
    if hp[i]==0:
        i+=1
    while i<len(hp) and j<len(tp) and mask[hp[i]]==1 and mask[tp[j]]==1:#还要小于mask
        if hp[i]<=tp[j]:
            pl = offset_map[hp[i]][0]
            pr = offset_map[tp[j]][1]
            all_entity.append((text[pl:pr],hp[i],tp[j]))
            i = i+1
            j = j+1
        else:
            j = j+1
    return all_entity

def from_sub_get_rel_and_obj(model,text_encode,sub_be,sub_ed,text,offset_map,mask,sub):
    sub_be_muti = multihot(len(mask),[sub_be])
    sub_ed_muti = multihot(len(mask),[sub_ed])

    sub_be_muti = torch.tensor(sub_be_muti).to(device).unsqueeze(0)
    sub_ed_muti = torch.tensor(sub_ed_muti).to(device).unsqueeze(0)

    obj_head_pre,obj_tail_pre = model.get_obj_pre(text_encode,sub_be_muti,sub_ed_muti)

    obj_head_pre, obj_tail_pre = obj_head_pre[0].T,obj_tail_pre[0].T
    sub_rel_obj = []
    for i in range(obj_head_pre.shape[0]):
        o_h = obj_head_pre[i].squeeze(-1)
        o_t = obj_tail_pre[i].squeeze(-1)
        o_h = torch.where(o_h>0.5)[0].tolist()
        o_t = torch.where(o_t>0.5)[0].tolist()
        if len(o_h)==0 or len(o_t)==0:
            continue
        all_obj = get_eneity(text,o_h,o_t,offset_map,mask)
        for obj in all_obj:
            sub_rel_obj.append((sub,index_2_rel[i],obj[0]))
    return sub_rel_obj


def report(model,texts,batch_pre_y,batch_triple_list,batch_mask,batch_offset_mapping,text_encode):
    all_pre = []
    correct_num, predict_num, gold_num = 0, 0, 0
    sub_head_pre, sub_tail_pre = batch_pre_y
    for i in range(sub_head_pre.shape[0]):
        #获取每个batch数据
        text = texts[i]
        single_sub_head = sub_head_pre[i]
        single_sub_tail = sub_tail_pre[i]
        mask = batch_mask[i]
        offset_mapping = batch_offset_mapping[i]
        triple_list = batch_triple_list[i]

        #获取sub的头位置
        single_sub_head = single_sub_head.squeeze(dim=-1)
        head_all_pos = torch.where(single_sub_head>0.5)[0].tolist()
        #获取sub的尾位置
        single_sub_tail = single_sub_tail.squeeze(dim=-1)
        tail_all_pos = torch.where(single_sub_tail > 0.5)[0].tolist()

        all_sub = get_eneity(text,head_all_pos,tail_all_pos,offset_mapping,mask)

        all_sub = all_sub[0:1]
        sub_rel_obj = []
        for sub_information in all_sub:
            sub_rel_obj.extend(from_sub_get_rel_and_obj(model,text_encode[i],sub_information[1],sub_information[2],text,offset_mapping,mask,sub_information[0]))

        all_pre.append(sub_rel_obj)
        # triple_list

        correct_num += len(set(sub_rel_obj) & set(triple_list))
        predict_num += len(sub_rel_obj)
        gold_num += len(set(triple_list))

    return all_pre,correct_num, predict_num, gold_num
if __name__ == "__main__":
    train_data = read_data(os.path.join('data2','duie_train.json'),200)
    dev_data = read_data(os.path.join('data2', 'duie_dev.json'), 200)
    rel_2_index,index_2_rel = build_rel_2_index()

    batch_size = 20
    epoch = 10
    model_name = '../data/bert_base_chinese'
    lr = 1e-5
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device="cpu"

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_dataset = C_Dataset(train_data,rel_2_index,tokenizer)
    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,collate_fn=train_dataset.operate_data)
    dev_dataset = C_Dataset(dev_data, rel_2_index, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size,
                                  collate_fn=dev_dataset.operate_data)

    model = Cas_Model(model_name,len(rel_2_index)).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    for e in range(epoch):
        model.train()
        # for batch_text,batch_mask,batch_sub,batch_sub_rnd,batch_obj_rel in tqdm(train_dataloader):
        #     input = (
        #         torch.tensor(batch_text['input_ids']).to(device),
        #         torch.tensor(batch_sub_rnd['heads_seq']).to(device),
        #         torch.tensor(batch_sub_rnd['tails_seq']).to(device),
        #
        #     )
        #     mask = torch.tensor(batch_mask).to(device)
        #     pre_y = model.forward(input,mask)
        #
        #     true_y = (
        #         torch.tensor(batch_sub['heads_seq']).to(device),
        #         torch.tensor(batch_sub['tails_seq']).to(device),
        #         torch.tensor(batch_obj_rel['heads_mx']).to(device),
        #         torch.tensor(batch_obj_rel['tails_mx']).to(device),
        #     )
        #     opt.zero_grad()
        #     loss = model.loss_fn(pre_y,true_y,mask)
        #     loss.backward()
        #     opt.step()
        correct_num, predict_num, gold_num = 0,0,0
        model.eval()
        for batch_text,batch_mask,batch_sub,batch_sub_rnd,batch_obj_rel in tqdm(dev_dataloader):
            input = (
                torch.tensor(batch_text['input_ids']).to(device),
            )
            batch_mask = torch.tensor(batch_mask).to(device)
            text_encode,batch_pre_y = model.forward(input, batch_mask)

            batch_triple_list = batch_text['triple_list']
            batch_offset_mapping = batch_text['offset_mapping']
            text = batch_text['text']
            all_pre,d_c,d_p,d_g = report(model,text,batch_pre_y,batch_triple_list,batch_mask,batch_offset_mapping,text_encode)

            correct_num += d_c
            predict_num += d_p
            gold_num += d_g
        precision = correct_num / (predict_num + 1e-8)
        reacall = correct_num / (gold_num + 1e-8)
        f1_score = 2 * precision * reacall / (precision + reacall + 1e-8)

        print(f"f1:{f1_score:.3f},predict_num:{predict_num},gold_num:{gold_num}")
