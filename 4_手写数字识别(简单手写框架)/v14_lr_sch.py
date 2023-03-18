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
        self.u = 0.2
        self.last_w_grad = 0
        self.last_b_grad = 0

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


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        return softmax(x)

    def backward(self, G):
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
        for p in params:
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
            Linear(784, 100),
            Tanh(),
            Linear(100, 50),
            Sigmoid(),
            Linear(50, 10),
            Softmax()
        ]

    def forward(self, x, labels=None):
        for layer in self.layers:
            x = layer.forward(x)
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
            G = layer.backward(G)

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


if __name__ == '__main__':
    np.random.seed(100)
    train_X = load_images(os.path.join('..', 'data', 'train-images.idx3-ubyte')) / 255.0
    train_Y = label2onehot(load_labels(os.path.join('..', 'data', 'train-labels.idx1-ubyte')))
    test_X = load_images(os.path.join('..', 'data', 't10k-images.idx3-ubyte')) / 255.0
    test_Y = label2onehot(load_labels(os.path.join('..', 'data', 't10k-labels.idx1-ubyte')))
    dataset = Dataset(batch_size=32, X=train_X, Y=train_Y)
    test_dataset = Dataset(batch_size=32, X=test_X, Y=test_Y)

    epoch = 100
    maxlr = 0.002
    minlr = 1e-6

    model = Model()
    opt = Adam(model.parameters())
    ttt = 0
    for e in range(epoch):
        lr = maxlr -(maxlr-minlr)/epoch*e
        opt.lr = lr
        print(f'e={e} lr={lr} {"*" * 100}')
        sum_loss, sum_acc = 0, 0
        batch_num = 0
        for x, labels in dataset:
            # forward
            x, loss = model(x, labels)
            # bachward
            model.backward()
            opt.step()
            ttt = ttt+1
            if ttt %3 ==0:
                opt.zero_grad()
            sum_loss += loss
            batch_num += 1
            sum_acc += np.sum(np.argmax(x, axis=1) == np.argmax(labels, axis=1))
            # print(f'batch_acc={np.sum(np.argmax(soft_pre,axis=1) == np.argmax(labels,axis=1))/images.shape[0]}')
        print(f'train_loss={sum_loss / batch_num:.10f}  train_acc={sum_acc / 60000:.10f} ')
        test_loss, test_acc = 0, 0
        batch_num = 0
        for X, Y in test_dataset:
            X = model(X)
            loss = -np.sum(Y * np.log(X)) / X.shape[0]
            test_loss += loss
            test_acc += np.sum(np.argmax(X, axis=1) == np.argmax(Y, axis=1))
            batch_num += 1
        print(f'test_loss={test_loss / batch_num:.10f}  test_acc={test_acc / 10000:.10f} ')
