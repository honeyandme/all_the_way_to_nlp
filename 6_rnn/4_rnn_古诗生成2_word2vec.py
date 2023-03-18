import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gensim.models
import os
import random
from tqdm import tqdm
def get_data(path):
    with open(path,"r") as f:
        all_data = f.read().split('\n')
    return all_data
def train_word2vec(all_data,embedding_num):
    vec_savepath = os.path.join('..', 'data', 'rnn_data', '5_word2vec.pt')
    if os.path.exists(vec_savepath):
        model = gensim.models.Word2Vec.load(vec_savepath)
        if model.wv.vector_size==embedding_num:
            return model

    word2vec = gensim.models.Word2Vec(all_data,vector_size=embedding_num,window=5,min_count=1,workers=5)
    word2vec.save(vec_savepath)
    return word2vec
class poetry_dataset(Dataset):
    def __init__(self,all_data,word2vec):
        self.all_data = all_data
        self.word2vec = word2vec
    def __getitem__(self, x):
        x = all_data[x]
        x_emb = [torch.tensor(self.word2vec.wv[k]) for k in x]
        label = [self.word2vec.wv.key_to_index[k] for k in x]
        return torch.stack(x_emb),torch.tensor(label)
    def __len__(self):
        return len(all_data)
class gen_model(nn.Module):
    def __init__(self,corp_len,embedding_num,hidden_size):
        super(gen_model, self).__init__()
        self.crop_len = corp_len
        self.rnn = nn.RNN(embedding_num,hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.V = nn.Linear(hidden_size,corp_len)
        self.dropout = nn.Dropout(0.15)
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x,label = None,hs_0 = None):
        if label is not None:
            o, t = self.rnn(x)
            o = self.V(o)
            loss = self.loss_fn(o.reshape(-1,self.crop_len),label.reshape(-1))
            return loss
        o, t = self.rnn(x,hs_0)
        o = self.dropout(o)
        o = self.V(o)
        return torch.argmax(o),t
def auto_gen_poetry():
    global word2vec,model,config
    letter = random.choice(word2vec.wv.index_to_key)
    result = [letter]
    hs_0 = torch.zeros((1,1,config["hidden_size"]))
    for i in range(23):
        idx,hs_0 = model(torch.tensor(word2vec.wv[letter].reshape((1,1,-1))),hs_0=hs_0)
        letter = word2vec.wv.index_to_key[idx]
        result.append(letter)
    return "".join(result)
if __name__ == "__main__":
    config = {
        "epoch": 100,
        "lr": 0.002,
        "batch_size": 20,
        "embedding_num": 125,
        "hidden_size":128,
    }
    all_data = get_data(os.path.join('..', 'data','rnn_data','poetry_5.txt'))
    word2vec = train_word2vec(all_data,config["embedding_num"])
    train_ds = poetry_dataset(all_data,word2vec)
    train_dl = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    corp_len = len(word2vec.wv)

    model = gen_model(corp_len=corp_len,embedding_num=config["embedding_num"],hidden_size=config["hidden_size"])
    opt = torch.optim.Adam(model.parameters(),lr = config["lr"])
    #config["corp_len"] =
    for e in range(config["epoch"]):
        loss_sum = 0
        ba_n = 0
        for x,y in tqdm(train_dl):
            opt.zero_grad()
            x_input = x[:,:-1,:]
            y_input = y[:,1:]
            loss = model(x_input,label=y_input)
            loss_sum += loss
            ba_n+=1
            loss.backward()
            opt.step()
        loss_sum /= ba_n
        print(f'e={e} loss_sum={loss_sum:.8f}')
        #print(auto_gen_poetry())