import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
if __name__=='__main__':
    path = os.path.join('..','data','上海二手房价.csv')
    data1 = pd.read_csv(path)
    fu_year = data1['建成年份'].values.reshape(-1, 1)
    pri_year = MinMaxScaler()
    pri_year.fit(fu_year)
    year = pri_year.transform(fu_year)

    fu_floor = data1['楼层'].values.reshape(-1, 1)
    pri_floor = MinMaxScaler()
    pri_floor.fit(fu_floor)
    floor = pri_floor.transform(fu_floor)

    fu_mian = data1['面积（平米）'].values.reshape(-1, 1)
    pri_mian = MinMaxScaler()
    pri_mian.fit(fu_mian)
    mian = pri_mian.transform(fu_mian)

    fu_price = data1['房价（元/平米）'].values.reshape(-1,1)
    pri_price = MinMaxScaler()
    pri_price.fit(fu_price)
    price = pri_price.transform(fu_price)


    features = np.stack((mian,floor,year),axis=-1).squeeze(axis=1)


    k = np.random.normal(0,0.5,(features.shape[1],1))
    epoch = 100
    lr = 0.1
    for e in range(epoch):
        pre = features @ k
        loss = np.mean((pre-price)**2)

        G = (pre-price)/pre.shape[0]

        delta_k = features.T @ G

        k-= lr*delta_k

        print(f'loss={loss:.2f}')



