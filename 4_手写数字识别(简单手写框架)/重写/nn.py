import numpy as np
from parameter import Parameter
from utils import sigmoid,softmax



class Module():
    def __init__(self):
        self.info = self.__class__.__name__
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Sigmoid(Module):
    def forward(self,x):
        r = sigmoid(x)
        self.r = r
        return r
    def backward(self,G):
        return G*self.r*(1-self.r)

class Tanh(Module):
    pass
class ReLU(Module):
    pass
class PReLU(Module):
    pass
class Softmax(Module):
    def forward(self,x):
        return softmax(x)
    def backward(self,G):
        return G
class Linear(Module):
    def __init__(self,in_feature,out_feature):
        super(Linear, self).__init__()
        self.w = Parameter(np.random.normal(0,0.5,(in_feature,out_feature)))
        self.b = Parameter(np.random.normal(0,0.5,(1,out_feature)))
    def forward(self,x):
        self.A = x
        x = x @ self.w.weight + self.b.weight
        return x
    def backward(self,G):
        self.w.grad += self.A.T @ G
        self.b.grad += np.sum(G,axis=0,keepdims=True)
        return G @ self.w.weight.T

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers=[
            Linear(784,100),
            Sigmoid(),
            Linear(100,10),
            Softmax()
        ]
    def forward(self,x,label=None):
        for layer in self.layers:
            x = layer(x)
        if label is not None:
            loss = -np.sum(label * np.log(x))/x.shape[0]
            self.G = x-label
            return x,loss
        return x
    def backward(self):
        G = self.G
        for layer in self.layers[::-1]:
            G = layer.backward(G)
        return G
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
    def __repr__(self):
        lis = []
        for layer in self.layers:
            if(layer.info=='Linear'):
                lis.append(f'{layer.info}({layer.w.weight.shape[0]},{layer.w.weight.shape[1]})')
            lis.append(layer.info)
        return '\n'.join(lis)
