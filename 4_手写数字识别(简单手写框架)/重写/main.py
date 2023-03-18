from utils import load_labels,load_images,label2onehot
import utils
import numpy as np
import os
from DataSet import Dataset
from nn import Model
from optim import MSGD,Adam
if __name__ == '__main__':
    np.random.seed(100)
    train_X = load_images(os.path.join('data', 'train-images.idx3-ubyte')) / 255.0
    train_Y = label2onehot(load_labels(os.path.join('data', 'train-labels.idx1-ubyte')))
    test_X = load_images(os.path.join('data', 't10k-images.idx3-ubyte')) / 255.0
    test_Y = label2onehot(load_labels(os.path.join('data', 't10k-labels.idx1-ubyte')))

    train_dataset = Dataset(32,train_X,train_Y)
    test_dataset = Dataset(32,test_X,test_Y)

    model = Model()
    opt = Adam(model.parameters())
    epoch = 100
    lr = 0.01

    for e in range(epoch):
        print(f'e={e}  {"*"*100}')
        loss_sum,batch = 0,0
        acc = 0
        for X,label in train_dataset:
            opt.zero_grad()
            pre,loss = model(X,label)
            model.backward()
            opt.step()
            loss_sum+=loss
            batch+=1
            acc += np.sum(np.argmax(pre,axis=1)==np.argmax(label,axis=1))
        print(f'train loss={loss_sum/batch:.10f} acc={acc/60000:.10f} {acc}')
        loss_sum, batch = 0, 0
        acc = 0
        for X,label in test_dataset:
            pre,loss = model(X,label)
            loss_sum+=loss
            batch+=1
            acc += np.sum(np.argmax(pre,axis=1)==np.argmax(label,axis=1))
        print(f'test loss={loss_sum/batch:.10f} acc={acc/10000:.10f} {acc}')
