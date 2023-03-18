import numpy as np

def sigmoid(x):
    res =  1/(1+np.exp(-x))
    res = np.clip(res,1e-10,0.9999999999)
    return res


if __name__ == "__main__":
    np.random.seed(100)
    dogs = np.array([[8.9,12],[9,11],[10,13],[9.9,11.2],[12.2,10.1],[9.8,13],[8.8,11.2]],dtype = np.float32)   # 0
    cats = np.array([[3,4],[5,6],[3.5,5.5],[4.5,5.1],[3.4,4.1],[4.1,5.2],[4.4,4.4]],dtype = np.float32)
    label = np.array([0]*7+[1]*7,dtype=np.int32).reshape(-1,1)
    X = np.vstack((dogs,cats))

    w = np.random.normal(0,0.5,(X.shape[1],1))
    b = 0
    lr = 0.07
    epoch = 1000
    for e in range(epoch):
        pre = X @ w + b
        pre_s = sigmoid(pre)
        loss = -np.mean(label*np.log(pre_s)+(1-label)*np.log(1-pre_s))
        G = (pre_s-label)/14
        delta_w = (X.T @ G)
        delta_b = np.sum(G)
        w-=lr*delta_w
        b-=delta_b

    print(f'loss={loss:.5f}')
    while(True):
        x1 = float(input('请输入第一个特征:'))
        x2 = float(input('请输入第二个特征:'))
        pre = np.array([[x1,x2]]) @ w+b
        if(pre<0):
            print('dog')
        else :
            print('cat')
