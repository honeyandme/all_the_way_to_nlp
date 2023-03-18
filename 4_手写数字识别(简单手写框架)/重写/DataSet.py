import numpy as np
class Dataset():
    def __init__(self,batch_size,X,Y=None,shuffle=True):
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        self.batch_size = batch_size
    def __iter__(self):
        return Dataloader(self)
    def __len__(self):
        return self.X.shape[0]

class Dataloader():
    def __init__(self,dataset):
        self.dataset = dataset
        self.it_idx = np.arange(len(self.dataset))
        if self.dataset.shuffle:
            np.random.shuffle(self.it_idx)
        self.cursor = 0
    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        idx = self.it_idx[self.cursor:self.cursor+self.dataset.batch_size]
        self.cursor += self.dataset.batch_size
        return self.dataset.X[idx],self.dataset.Y[idx]

if __name__ == "__main__":
    x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    y = np.array([1,2,3,4])
    dataset = Dataset(3,x,y)
    for x,y in dataset:
        print(x,y)
