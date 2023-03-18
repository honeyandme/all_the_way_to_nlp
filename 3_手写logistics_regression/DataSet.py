import numpy as np
class Dataset():
    def __init__(self,X,Y,batch_size,shuffle=True,processing=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert len(X) == len(Y)
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
        return self.dataset.X[idx,:],self.dataset.Y[idx,:]