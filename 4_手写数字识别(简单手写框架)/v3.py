import numpy as np
import struct
import os
import matplotlib.pyplot as plt

def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)
class Dataset():
    def __init__(self,batch_size,X,Y=None,shuffle=True,processing=True):
        self.X = X
        if Y is not None:
            self.Y = Y
            assert len(X) == len(Y)
        else :
            self.Y = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        if processing:
            pass
    def __len__(self):
        return len(self.X)
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
        if self.dataset.Y is not None:
            return self.dataset.X[idx,:],self.dataset.Y[idx,:]
        else:return self.dataset.X[idx, :]

def label2onehot(labels,class_num=10):
    num = labels.shape[0]
    one_hot_label = np.zeros((num,class_num))
    for i,x in enumerate(labels):
        one_hot_label[i][x] = 1
    return one_hot_label
def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex,axis=1,keepdims=True)
    return ex/sum_ex
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    res = np.clip(res, 1e-10, 0.99999999)
    return res
def tanh(x):
    return 2*sigmoid(2*x)-1
class NN_layer:
    pass
class Linear(NN_layer):
    def __init__(self,in_feature,out_feature):
        self.W = np.random.normal(0,0.5,(in_feature,out_feature))
        self.b = np.random.normal(0,0.5,(1,out_feature))
    def forward(self,x):
        self.A = x
        return x@self.W+self.b
    def backward(self,G):
        delta_w = self.A.T @ G
        self.W -= lr*delta_w
        delta_b = np.sum(G,axis=0,keepdims=True)
        self.b-=lr*delta_b
        return G @ self.W.T
if __name__ =='__main__':
    train_X = load_images(os.path.join('..','data','train-images.idx3-ubyte'))/255.0
    train_Y = label2onehot(load_labels(os.path.join('..','data','train-labels.idx1-ubyte')))
    test_X =load_images(os.path.join('..','data','t10k-images.idx3-ubyte'))/255.0
    test_Y =label2onehot(load_labels(os.path.join('..','data','t10k-labels.idx1-ubyte')))
    dataset = Dataset(batch_size=32,X=train_X,Y=train_Y)
    test_dataset = Dataset(batch_size=32,X=test_X,Y=test_Y)




    epoch = 100
    lr = 0.01
    linear1 = Linear(784,100)
    linear2 = Linear(100,10)


    for e in range(epoch):
        print(f'e={e}  {"*"*100}')
        sum_loss,sum_acc = 0,0
        batch_num = 0
        for images,labels in dataset:
            #forward
            H = linear1.forward(images)
            tanh_H = tanh(H)
            pre = linear2.forward(tanh_H)
            soft_pre = softmax(pre)

            #cal_loss
            loss = -np.sum(labels*np.log(soft_pre))/images.shape[0]
            #backward
            G = (soft_pre-labels)/images.shape[0] #L/pre
            delta_tanhH = linear2.backward(G)
            G2 = delta_tanhH*(1-tanh(H)**2)
            linear1.backward(G2)

            sum_loss += loss
            batch_num += 1
            sum_acc +=  np.sum(np.argmax(soft_pre,axis=1) == np.argmax(labels,axis=1))
            #print(f'batch_acc={np.sum(np.argmax(soft_pre,axis=1) == np.argmax(labels,axis=1))/images.shape[0]}')
        print(f'train_loss={sum_loss / batch_num:.10f}  train_acc={sum_acc / 60000:.10f} ')
        test_loss,test_acc = 0,0
        batch_num = 0
        for X,Y in test_dataset:
            H = linear1.forward(X)
            H = tanh(H)
            pre = linear2.forward(H)
            soft_pre = softmax(pre)
            loss = -np.sum(Y * np.log(soft_pre)) / X.shape[0]
            test_loss += loss
            test_acc +=np.sum(np.argmax(soft_pre,axis=1) == np.argmax(Y,axis=1))
            batch_num+=1
        print(f'test_loss={test_loss / batch_num:.10f}  test_acc={test_acc / 10000:.10f} ')
