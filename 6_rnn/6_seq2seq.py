import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
def get_data():
    en = ['aaa','obvious','how','capacity','visualize','object','numerical','patterns','answer','questions','some']
    ch = ['啊','明显的','怎么','能力','想象','物体','数字的','模式','回答','问题','一些']
    return en,ch
def build_word_to_idx(en,ch):
    en2idx = {'PAD':0,'UNK':1}
    ch2idx = {'PAD':0,'UNK':1,'B':2,'E':3}
    for word in en:
        for w in word:
            en2idx[w] = en2idx.get(w,len(en2idx))
    for word in ch:
        for w in word:
            ch2idx[w] = ch2idx.get(w,len(ch2idx))
    return en2idx,ch2idx
class seq_dataset(Dataset):
    def __init__(self,en,ch,config):
        self.en = en
        self.ch = ch
        self.config = config
        assert len(en)==len(ch)
    def __getitem__(self, x):
        en_x = [self.config["en2idx"][w] for w in en[x]]
        #en_x = en_x + [0]*(self.config["en_max_len"]-len(en_x))

        ch_x = [self.config["ch2idx"][w] for w in ch[x]]
        ch_x = [2]+ch_x+[3] #+[0]*(self.config["ch_max_len"]-len(ch_x))

        return en_x,ch_x,len(en_x),len(ch_x)
    def collate_fn(self,x):
        en_max_len = 0
        ch_max_len = 0
        for d in x:
            en_max_len = max(en_max_len,d[2])
            ch_max_len = max(ch_max_len, d[3])
        en,ch,en_len,ch_len=[],[],[],[]
        for d in x:
            en.append(d[0]+[0]*(en_max_len-len(d[0])))
            ch.append(d[1]+[0] * (ch_max_len - len(d[1])))
            en_len.append(d[2])
            ch_len.append(d[3])
        return torch.tensor(en),torch.tensor(ch),torch.tensor(en_len),torch.tensor(ch_len)

    def __len__(self):
        return len(self.en)
class seq2seq(nn.Module):
    def __init__(self,config):
        super(seq2seq, self).__init__()
        self.config = config
        self.en_embedding = nn.Embedding(config["en_crop"],config["embedding_num"])
        self.ch_embedding = nn.Embedding(config["ch_crop"], config["embedding_num"])
        self.encoder = nn.GRU(input_size=config["embedding_num"],hidden_size=config["hidden_num"],batch_first=True,bidirectional=False)
        self.decoder = nn.GRU(input_size=config["embedding_num"],hidden_size=config["hidden_num"],batch_first=True,bidirectional=False)
        self.classifier = nn.Linear(config["hidden_num"],config["ch_crop"])
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self,batch_en,batch_ch,en_len,ch_len):
        en_emb = self.en_embedding(batch_en)
        ch_emb = self.ch_embedding(batch_ch[:,:-1])

        pack = pack_padded_sequence(en_emb,en_len,batch_first=True,enforce_sorted=False)
        _,o = self.encoder(pack)
        #unpack,_ = pad_packed_sequence(_,batch_first=True)
        out1,out2 = self.decoder(ch_emb,o)
        pre = self.classifier(out1)
        loss = self.loss_fn(pre.reshape(-1,self.config["ch_crop"]),batch_ch[:,1:].reshape(-1))
        return loss

    def transfer(self,en,ch):
        en_emb = self.en_embedding(en)
        result = []
        _, o = self.encoder(en_emb)
        tot = 0
        while True:
            ch_emb = self.ch_embedding(ch)
            out1,o = self.decoder(ch_emb,o)
            ch = torch.argmax(self.classifier(o),dim=-1)
            tot = tot+1
            if int(ch)==3 or tot==20:
                break
            result.append(self.config["idx2ch"][ch])
        return "".join(result)

if __name__ == '__main__':
    en,ch = get_data()
    en2idx,ch2idx = build_word_to_idx(en,ch)
    config = {
        "en2idx":en2idx,
        "ch2idx":ch2idx,
        "idx2ch":list(ch2idx),
        "epoch":100,
        "batch_size":2,
        "embedding_num":100,
        "hidden_num":150,
        "en_crop":len(en2idx),
        "ch_crop":len(ch2idx),
        "lr":1e-3
    }
    dataset  = seq_dataset(en,ch,config)
    dataloader = DataLoader(dataset,batch_size=config["batch_size"],shuffle=False,collate_fn=dataset.collate_fn)
    model = seq2seq(config)
    opt = torch.optim.Adam(model.parameters(),lr = config["lr"])
    for e in range(config["epoch"]):
        loss_sum = 0
        for batch_en,batch_ch,en_len,ch_len in dataloader:
            opt.zero_grad()
            loss = model(batch_en,batch_ch,en_len,ch_len)
            loss.backward()
            opt.step()
            loss_sum+=loss
        print(f'e={e} loss_sum={loss_sum:.5f}')
    while True:
        x = input('请输入单词')
        x=[en2idx.get(w,1) for w in x]
        #x = x+[0]*(config["en_max_len"]-len(x))
        x_index = torch.tensor([x])
        ch = torch.tensor([[2]])
        pre = model.transfer(x_index,ch)
        print('pre='+pre)
