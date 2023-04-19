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
    all_text,all_tag = [],[]
    for data in all_data:
        data = eval(data)
        ent1,pos1 = data['h']['name'],data['h']['pos']
        ent2,pos2 = data['t']['name'],data['t']['pos']
        relation = data['relation']
        text = data['text']

        tag = ['O']*len(text)
        tag[pos1[0]:pos1[1]] = ['B-'+relation] +['I-'+relation]*(len(ent1)-1)
        tag[pos2[0]:pos2[1]] = ['B-' + relation] + ['I-' + relation] * (len(ent2) - 1)
        text = list(text)
        all_text.append(text)
        all_tag.append(tag)
    return all_text,all_tag


def build_tag_2_index(all_tag):
    mp = {'O':0,'unknown':1}
    for tag in all_tag:
        for word in tag:
            mp[word] = mp.get(word,len(mp))
    return mp


class Ndataset(Dataset):
    def __init__(self,all_text,all_tag,tokenizer,tag_2_index,max_len):
        self.all_text = all_text
        self.all_tag = all_tag
        self.tokenizer = tokenizer
        self.tag_2_index = tag_2_index
        self.max_len = max_len
    def __getitem__(self, x):
        text,tag = self.all_text[x][:self.max_len-2],self.all_tag[x][:self.max_len-2]
        text_idx = self.tokenizer.encode(text,add_special_tokens=True)
        tag_idx = [0]+[self.tag_2_index[i] for i in tag]+[0]

        text_idx += [0]*(self.max_len-len(text_idx))
        tag_idx += [0] * (self.max_len - len(tag_idx))
        assert len(text_idx)==len(tag_idx)
        return torch.tensor(text_idx),torch.tensor(tag_idx),len(tag)
    def __len__(self):
        return len(self.all_text)

class Bmodel(nn.Module):
    def __init__(self,model_name,tag_num):
        super(Bmodel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768,tag_num)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,x,y=None):
        bert_out,_ = self.bert(x,return_dict=False)
        pre = self.classifier(bert_out)

        if y is not None:
            loss = self.loss_fn(pre.reshape(-1,pre.shape[-1]),y.reshape(-1))
            return loss
        return torch.argmax(pre,dim=-1)
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        # torch.cuda.manual_seed(seed)  # 为当前GPU设置
        # torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
        torch.mps.manual_seed(seed)
        torch.mps.manual_seed_all(seed)
    #np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
if __name__ == "__main__":
    same_seeds(3407)
    train_text,train_tag = get_data(os.path.join('data','train_small.jsonl'))
    dev_text, dev_tag = get_data(os.path.join('data', 'val_small.jsonl'))
    tag_2_index = build_tag_2_index(train_tag)
    index_2_tag = list(tag_2_index)

    batch_size = 50
    model_name = '../data/bert_base_chinese'
    epoch = 3
    max_len = 50
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lr = 1e-5

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = Ndataset(train_text,train_tag,tokenizer,tag_2_index,max_len)
    train_dataloader = DataLoader(train_dataset,shuffle=False,batch_size=batch_size)

    dev_dataset = Ndataset(dev_text, dev_tag, tokenizer, tag_2_index, max_len)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size)

    model = Bmodel(model_name,len(tag_2_index)).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    for e in range(epoch):
        for x,y,batch_len in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            loss = model(x,y)
            loss.backward()
            opt.step()
        all_pre,all_tag = [],[]
        for x,y,batch_len in tqdm(dev_dataloader):
            x = x.to(device)
            pre = model(x)

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
        print(f"f1_score={f1}")
