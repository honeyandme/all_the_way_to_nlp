import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
class Dataset():
    def __init__(self,year,price,batch_size,shuffle=True):
        self.year = year
        self.price = price
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.year)
    def __iter__(self):
        return Dataloader(self)
class Dataloader():
    def __init__(self,dataset):
        self.dataset = dataset
        self.index = np.arange(len(dataset))
        if self.dataset.shuffle:
            np.random.shuffle(self.index)
        self.cursor = 0
    def __next__(self):
        if self.cursor>=len(self.dataset):
            raise StopIteration
        idx = self.index[self.cursor:self.cursor+self.dataset.batch_size]
        self.cursor+=self.dataset.batch_size
        return self.dataset.year[idx],self.dataset.price[idx]
def get_data():

    pri_year = MinMaxScaler()
    fu_year = np.array(range(1,101)).reshape(-1,1)
    pri_year.fit(fu_year)
    year = pri_year.transform(fu_year)
    fu_price = fu_year*700.2+np.random.normal(0,100,year.shape)
    assert len(fu_year)==len(fu_price)
    pri_price = MinMaxScaler()
    pri_price.fit(fu_price)
    price = pri_price.transform(fu_price)
    return year.reshape(-1),price.reshape(-1),pri_year,pri_price

if __name__=='__main__':
    #参数初始化
    k=1.0
    b=-1.0
    lr = 0.7
    epoch = 10
    batch_size = 5
    #获取数据集
    year,price,pri_year,pri_price = get_data()
    #开始训练
    for e in range(epoch):
        dataset = Dataset(year,price,batch_size)
        for batch_year,batch_price in dataset:
            #预测
            pre = k*batch_year+b
            #损失函数
            loss = np.mean((pre-batch_price)**2)

            #计算梯度
            delta_k = np.mean((pre-batch_price)*batch_year)
            delta_b = np.mean((pre-batch_price))

            #梯度下降
            k-=lr*delta_k
            b-=lr*delta_b

            print(f'loss={loss:.3f}')
    plt.plot(year,price,"r.")
    plt.plot([0,1],[b,k+b],"g-")
    plt.show()
    while True:
        x = int(input("请输入要预测的年份:"))
        x = pri_year.transform([[x]])
        pre = k*x+b
        pre = pri_price.inverse_transform(pre).tolist()[0][0]
        print(f'预测的结果为{pre:.3f}')