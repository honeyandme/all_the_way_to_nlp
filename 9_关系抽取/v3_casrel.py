import os
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
    def __len__(self):
        return len(self.all_data)

if __name__ == "__main__":
    train_data = read_data(os.path.join('data2','duie_dev.json'))
    rel_2_index,index_2_rel = build_rel_2_index()

    batch_size = 2
    epoch = 1
    model_name = '../data/bert_base_chinese'

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_dataset = C_Dataset(train_data,rel_2_index,tokenizer)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size)

    for e in range(epoch):
        for x,y in train_dataloader:
            pass