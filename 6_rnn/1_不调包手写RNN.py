import numpy as np
import gensim
import os
from tqdm import tqdm
def get_data(path,num=None):
    with open(path,"r") as f:
        all_data = f.read().split('\n')
    if num is not None:
        return all_data[:num]
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
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    #res = np.clip(res, 1e-10, 0.99999999)
    return res
def tanh(x):
    return 2 * sigmoid(2 * x) - 1
def softmax(x):
    x = np.exp(x)
    e_sum = np.sum(x,axis=1,keepdims=True)
    x/=e_sum
    return x
if __name__ == "__main__":
    np.random.seed(100)
    epoch = 10
    embedding_num = 128
    hidden_num  = 100
    lr = 0.01
    all_data = get_data('../data/rnn_data/poetry_5.txt')
    word2vec = train_word2vec(all_data,embedding_num=embedding_num)

    crop_len = len(word2vec.wv)

    W = np.random.normal(0,1/np.sqrt(embedding_num),(embedding_num,hidden_num))
    V = np.random.normal(0, 1 / np.sqrt(embedding_num), (hidden_num, crop_len))
    U = np.random.normal(0, 1 / np.sqrt(hidden_num), (hidden_num, hidden_num))
    bias_W = np.random.normal(0,0.2,(1,hidden_num))
    bias_V = np.random.normal(0, 0.2, (1, crop_len))
    bias_U = np.random.normal(0, 0.2, (1, hidden_num))


    for e in range(epoch):
        loss_sum, ba = 0, 0
        for sen in tqdm(all_data):
            cache = []
            x_emb = np.array([word2vec.wv[t] for t in sen[:-1]])
            y_label = [word2vec.wv.key_to_index[t] for t in sen[1:]]
            y_label_onehot = np.zeros((x_emb.shape[0],crop_len))

            for i,l in enumerate(y_label):
                y_label_onehot[i][l] = 1
            lasth__ = np.zeros((1, hidden_num))
            for x,y in zip(x_emb,y_label_onehot):
                x,y = x[None],y[None]
                h = x @ W + bias_W
                lasth__ = lasth__ @ U +bias_U
                h_ = h + lasth__
                h__  = tanh(h_)
                pre = h__@ V + bias_V
                lasth__ = h__
                so_pre = softmax(pre)
                loss_sum += -np.sum(y*np.log(so_pre))
                ba+=1
                cache.append((x,y,so_pre,h__,h_))

            last_delta_h_ = np.zeros((1,hidden_num))
            DV,DU,DW,bV,bU,bW = 0,0,0,0,0,0
            for x,label,so_pre,h__,h_ in cache[::-1]:
                G = (so_pre-label)/23
                DV +=  h__.T @ G

                delta_h__ = G @ V.T + last_delta_h_ @ U.T

                DU += h__.T @ last_delta_h_

                delta_h_ = delta_h__*(1-tanh(h_)**2)

                DW+=x.T @ delta_h_


                last_delta_h_ = delta_h_

                bV += np.sum(G, axis=0, keepdims=True)
                bU += np.sum(last_delta_h_,axis=0, keepdims=True)
                bW += np.sum(delta_h_,axis=0,keepdims=True)
            W -= lr * DW
            U -= lr * DU
            V -= lr * DV
            bias_W -= lr*bW
            bias_V -= lr*bV
            bias_U -= lr*bU
        print(f'e={e} loss={loss_sum / ba:.5f}')