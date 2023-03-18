import numpy as np
import struct
import os
import matplotlib.pyplot as plt
import pickle

import torch


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
    def __init__(self, batch_size, X, Y=None, shuffle=True, processing=True):
        self.X = X
        if Y is not None:
            self.Y = Y
            assert len(X) == len(Y)
        else:
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
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = np.arange(len(dataset))
        if self.dataset.shuffle:
            np.random.shuffle(self.index)
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        idx = self.index[self.cursor:self.cursor + self.dataset.batch_size]
        self.cursor += self.dataset.batch_size
        if self.dataset.Y is not None:
            return self.dataset.X[idx, :], self.dataset.Y[idx, :]
        else:
            return self.dataset.X[idx, :]


def label2onehot(labels, class_num=10):
    num = labels.shape[0]
    one_hot_label = np.zeros((num, class_num))
    for i, x in enumerate(labels):
        one_hot_label[i][x] = 1
    return one_hot_label


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    return ex / sum_ex


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    res = np.clip(res, 1e-10, 0.99999999)
    return res


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


class Module:
    def __init__(self):
        self.info = self.__class__.__name__

    # def __call__(self, x):
    #     return self.forward(x)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_feature, out_feature):
        super(Linear, self).__init__()
        self.info += f'({in_feature},{out_feature})'
        self.W = Parameter(np.random.normal(0, 0.5, (in_feature, out_feature)))
        self.b = Parameter(np.random.normal(0, 0.5, (1, out_feature)))

    def forward(self, x):
        self.A = x
        return x @ self.W.weight + self.b.weight

    def backward(self, G):
        self.W.grad += self.A.T @ G
        self.b.grad += np.sum(G, axis=0, keepdims=True)

        return G @ self.W.weight.T


class Parameter():
    def __init__(self, w):
        self.weight = w
        self.grad = np.zeros_like(w)


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        r = tanh(x)
        self.r = r
        return r

    def backward(self, G):
        return G * (1 - self.r ** 2)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        r = sigmoid(x)
        self.r = r
        return r

    def backward(self, G):
        return G * (self.r * (1 - self.r))
class ReLU(Module):
    def forward(self,x):
        self.neg = x<0
        x[self.neg] = 0
        return x
    def backward(self,G):
        G[self.neg] = 0
        return G
class PReLU(Module):
    def __init__(self,p=0.25):
        super(PReLU, self).__init__()
        self.p = p
    def forward(self,x):
        self.neg = x<0
        x[self.neg] *= self.p
        return x
    def backward(self,G):
        G[self.neg] *=self.p
        return G
class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        return softmax(x)

    def backward(self, G):
        return G

class Dropout(Module):
    def __init__(self,p=0.1):
        super(Dropout, self).__init__()
        self.p = p
    def forward(self,x,is_train):
        if is_train:
            tmp = np.random.rand(*x.shape)
            self.tmp = tmp<self.p
            x[self.tmp] = 0
        return x
    def backward(self,G,is_train):
        if is_train:
            G[self.tmp] = 0
        return G
class Optim():
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr
    def zero_grad(self):
        for p in self.params:
            p.grad = 0


class SGD(Optim):

    def step(self):
        for p in self.params:
            p.weight -= self.lr * p.grad


class MSGD(Optim):
    def __init__(self, params, lr=0.1):
        super(MSGD, self).__init__(params, lr)
        self.u = 0.1
        for p in self.params:
            p.last_grad = 0

    def step(self):
        for p in self.params:
            p.weight -= self.lr * (self.u * p.last_grad + (1 - self.u) * p.grad)
            p.last_grad = p.grad


class Adam(Optim):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, e=1e-8):
        super(Adam, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = e
        for p in self.params:
            p.m = 0
            p.v = 0
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.params:
            gt = p.grad
            p.m = self.beta1 * p.m + (1 - self.beta1) * gt
            p.v = self.beta2 * p.v + (1 - self.beta2) * gt * gt
            mt_ = p.m / (1 - self.beta1 ** self.t)
            vt_ = p.v / (1 - self.beta2 ** self.t)
            p.weight -= self.lr * mt_ / (np.sqrt(vt_) + self.e)


class Model(Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            Linear(784, 256),
            Tanh(),
            Dropout(0.1),
            Linear(256, 128),
            Tanh(),
            Dropout(0.1),
            Linear(128, 10),
            Softmax()
        ]
        self.is_train = False
    def forward(self, x, labels=None):
        for layer in self.layers:
            if layer.info == 'Dropout':
                x = layer.forward(x,self.is_train)
            else:x = layer.forward(x)
        if labels is not None:
            # cal_loss
            loss = -np.sum(labels * np.log(x)) / x.shape[0]
            self.G = (x - labels) / x.shape[0]  # L/pre
            return x, loss
        else:
            return x

    def backward(self):
        G = self.G
        for layer in self.layers[::-1]:
            if layer.info=='Dropout':
                G = layer.backward(G,self.is_train)
            else:G = layer.backward(G)

    def __repr__(self):
        infos = []
        for layer in self.layers:
            infos.append(layer.info)
        return '\n'.join(infos)

    def parameters(self):
        res = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                res.append(value)
            elif "__iter__" in dir(value):
                for val in value:
                    try:
                        val_dic = val.__dict__
                    except:
                        break
                    for p in val_dic.values():
                        if isinstance(p, Parameter):
                            res.append(p)
        return res
    def train(self):
        self.is_train = True
    def eval(self):
        self.is_train = False
import cv2
def get_test(dir_path):
    tu_lis = os.listdir(dir_path)

    tu_lis = [x for x in tu_lis if x[-3:]=='png']
    label = [int(x[:-4]) for x in tu_lis]
    all_img = []
    print(tu_lis)
    for bi,tu in enumerate(tu_lis):
        image = cv2.imread(os.path.join(dir_path,tu),cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(28,28))

        #cv2.imshow('hehe', image)
        cv2.imwrite(os.path.join(dir_path,'temp_jpg',f'{label[bi]}.jpg'),image)
        all_img.append(image.reshape(-1))

    return np.array(all_img,dtype=np.float32),np.array(label)
if __name__ == '__main__':
    np.random.seed(100)
    x,y = get_test(os.path.join('t2'))
    x/=255.0
    epoch = 100
    maxlr = 0.003
    minlr = 1e-6

    # model = Model()
    with open('best_model.pt','rb') as f:
        model = pickle.load(f)
    x = model(x)
    x = np.argmax(x,axis=-1)
    print(np.sum(x==y))
    print(x)
    print(y)
    # opt = Adam(model.parameters())
    # ttt = 0
    # best_acc = -1
    # for e in range(epoch):
    #     model.train()
    #     lr = maxlr -(maxlr-minlr)/epoch*e
    #     opt.lr = lr
    #     print(f'e={e} lr={lr} {"*" * 100}')
    #     sum_loss, sum_acc = 0, 0
    #     batch_num = 0
    #     for x, labels in dataset:
    #         # forward
    #         x, loss = model(x, labels)
    #         # bachward
    #         model.backward()
    #         opt.step()
    #         opt.zero_grad()
    #         sum_loss += loss
    #         batch_num += 1
    #         sum_acc += np.sum(np.argmax(x, axis=1) == np.argmax(labels, axis=1))
    #         # print(f'batch_acc={np.sum(np.argmax(soft_pre,axis=1) == np.argmax(labels,axis=1))/images.shape[0]}')
    #     sum_loss /= batch_num
    #     sum_acc /= len(dataset)
    #     print(f'train_loss={sum_loss:.10f}  train_acc={sum_acc:.10f} ')
    #
    #
    #     model.eval()
    #     test_loss, test_acc = 0, 0
    #     batch_num = 0
    #     for X, Y in test_dataset:
    #         X = model(X)
    #         loss = -np.sum(Y * np.log(X)) / X.shape[0]
    #         test_loss += loss
    #         test_acc += np.sum(np.argmax(X, axis=1) == np.argmax(Y, axis=1))
    #         batch_num += 1
    #     test_loss/=batch_num
    #     test_acc/=len(test_dataset)
    #     if test_acc>best_acc:
    #         best_acc = test_acc
    #         print(f'test_loss={test_loss:.10f}  test_acc={test_acc:.10f} ------------------------------>best')
    #         with open('best_model.pt','wb') as f:
    #             pickle.dump(model,f)
    #     else:
    #         print(f'test_loss={test_loss:.10f}  test_acc={test_acc:.10f} ')
