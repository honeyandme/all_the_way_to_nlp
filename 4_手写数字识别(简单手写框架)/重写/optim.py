import numpy as np
class optim():
    def __init__(self,params,lr=0.1):
        self.params=params
        self.lr = lr
    def zero_grad(self):
        for p in self.params:
            p.grad = 0
class SGD(optim):
    def step(self):
        for p in self.params:
            p.weight -= self.lr*p.grad

class MSGD(optim):
    def __init__(self,params,lr=0.1,u=0.2):
        super(MSGD, self).__init__(params,lr)
        self.u = u
        for p in self.params:
            p.last_grad = 0
    def step(self):
        for p in self.params:
            p.weight -= self.lr*(self.u*p.last_grad+(1-self.u)*p.grad)
            p.last_grad = p.grad
class Adam(optim):
    def __init__(self, params, lr=0.005, beta1=0.9, beta2=0.999, e=1e-8):
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