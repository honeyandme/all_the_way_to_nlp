import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import gensim.models
import os
import random
from tqdm import tqdm
from transformers import BertTokenizer,BertModel
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
    def __init__(self,all_data,word_2_vec,word2idx):
        self.all_data = all_data
        self.word_2_vec = word_2_vec
        self.word2idx = word2idx
    def __getitem__(self, x):
        x = all_data[x]
        x_emb = [torch.tensor(self.word_2_vec[word2idx[k]]) for k in x]
        label = [self.word2idx[k] for k in x]
        return torch.stack(x_emb),torch.tensor(label)
    def __len__(self):
        return len(all_data)
class gen_model(nn.Module):
    def __init__(self,corp_len,embedding_num=768,hidden_size=128):
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
    global word_2_vec,model,config,idx2word,word2idx
    letter = random.choice(idx2word)
    result = [letter]
    hs_0 = torch.zeros((1,1,config["hidden_size"]))
    for i in range(23):
        idx,hs_0 = model(word_2_vec[word2idx[letter]].reshape((1,1,-1)),hs_0=hs_0)
        letter = idx2word[idx]
        result.append(letter)
    return "".join(result)
def build_word2idx(all_data,vec):
    mp = {}
    word_2_vec = []
    with open(os.path.join('..','data','bert_base_chinese','vocab.txt'),'r') as f:
        all_words = f.read().split('\n')
        bert_word_2_index = {word:i for i,word in enumerate(all_words)}
    for sen in all_data:
        for w in sen:
            if w not in mp:
                mp[w] = len(mp)
                if w not in bert_word_2_index:
                    word_2_vec.append(torch.normal(0,0.2,(768,)))
                else:
                    word_2_vec.append(vec[bert_word_2_index[w]])
    return mp,torch.stack(word_2_vec)
if __name__ == "__main__":
    #bt = BertTokenizer.from_pretrained(os.path.join('..','data','bert_base_chinese'))

    #word_to_vec = torch.load(os.path.join('..','data','bertvec_torch.pt'))
    config = {
        "epoch": 100,
        "lr": 0.002,
        "batch_size": 20,
        "embedding_num": 125,
        "hidden_size":128,
    }
    all_data = get_data(os.path.join('..', 'data','rnn_data','poetry_5.txt'))
    bert = BertModel.from_pretrained(os.path.join('..', 'data', 'bert_base_chinese'))
    vec = bert.embeddings.word_embeddings.weight
    word2idx,word_2_vec = build_word2idx(all_data,vec)
    idx2word = list(word2idx)
    #5_word2vec = train_word2vec(all_data,config["embedding_num"])
    train_ds = poetry_dataset(all_data,word_2_vec,word2idx)
    train_dl = DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    corp_len = len(word2idx)

    model = gen_model(corp_len=corp_len)
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
        print(auto_gen_poetry())