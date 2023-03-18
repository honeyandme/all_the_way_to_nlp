import os
import numpy as np
import gensim
import re
import jieba

from sklearn.metrics.pairwise import cosine_similarity
def split_data(path):
    with open(path,'r',encoding='utf8') as f:
        all_data = f.read().split('\n')
    now_data = []
    # pat = '[\u4e00-\u9fa5]'
    for tc in all_data:
        # tc = re.findall(pat,data)
        # tc = "".join(tc)
        tc = tc.split("	")[0]
        tc = jieba.lcut(tc)
        now_data.append(tc)
    with open(os.path.join('data','train_split.txt'),'w') as f:
        result = [" ".join(i) for i in now_data]
        f.write("\n".join(result))
def train_word2vec():
    if os.path.exists(os.path.join('data','5_word2vec.model')):
        return gensim.models.Word2Vec.load(os.path.join('data','5_word2vec.model'))

    split_path = os.path.join('data','train_split.txt')
    if os.path.exists(split_path)==False:
        split_data(os.path.join('data','train.txt'))
    with open(split_path,'r',encoding='utf8') as f:
        train_data = f.read().split('\n')
    train_data = [x.split(' ') for x in train_data]
    model = gensim.models.Word2Vec(sentences=train_data,vector_size=100,window=7,min_count=1,sg=0,hs=1)
    model.save(os.path.join('data','5_word2vec.model'))
    return model
if __name__== '__main__':
    model = train_word2vec()
    while True:
        text1 = input('句子1:')
        text2 = input('句子2:')
        t1 = jieba.lcut(text1)
        t2 = jieba.lcut(text2)

        x1 = np.array([model.wv[x] for x in t1])
        x2 = np.array([model.wv[x] for x in t2])

        x1_mean = np.mean(x1,axis=0,keepdims=True)
        x2_mean = np.mean(x2, axis=0, keepdims=True)

        x1_max = np.max(x1, axis=0, keepdims=True)
        x2_max = np.max(x2, axis=0, keepdims=True)

        print(f"mean:{cosine_similarity(x1_mean,x2_mean)[0][0]:.3f} max:{cosine_similarity(x1_max,x2_max)[0][0]:.3f}")