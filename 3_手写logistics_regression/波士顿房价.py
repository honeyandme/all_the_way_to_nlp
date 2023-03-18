import numpy as np
import pandas as pd
from DataSet import Dataset

from sklearn.preprocessing import normalize,minmax_scale
if __name__=='__main__':


    np.random.seed(100)
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    Y = raw_df.values[1::2, 2].reshape(-1,1)

    X= minmax_scale(X,axis=0)
    #df = pd.DataFrame(X)
    Y = minmax_scale(Y,axis=0)

    dataset = Dataset(X,Y,batch_size=40)

    w = np.random.normal(0,0.5,(X.shape[1],1))
    b = 0
    lr = 0.1
    epoch = 500
    for e in range(epoch):
        sum_loss = 0
        batch_num = 0
        for x,y in dataset:
            pre = x @ w +b
            loss = np.mean((pre-y)**2)
            G = (pre-y)/x.shape[0]
            delta_w = x.T @ G
            delta_b = np.sum(G)

            w-=lr*delta_w
            b-=lr*delta_b
            sum_loss+=loss
            batch_num+=1

        print(f'平均loss={sum_loss/batch_num:.10f}')
