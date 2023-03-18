import numpy as np
class Parameter():
    def __init__(self,w):
        self.weight = w
        self.grad = np.zeros_like(w)