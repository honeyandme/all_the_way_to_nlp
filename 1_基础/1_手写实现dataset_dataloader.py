import numpy as np
class DataSet:
    def __init__(self,all_data,all_label,batch_size,shuffle=True):
        self.all_data = np.array(all_data)
        self.all_label = np.array(all_label)
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        return DataLoader(self)
    def __len__(self):
        return len(all_data)


class DataLoader:
    def __init__(self,dataset):
        self.dataset = dataset
        self.idxlist = np.arange(len(dataset))
        self.cursor = 0
        if self.dataset.shuffle:
            np.random.shuffle((self.idxlist))
    def __getitem__(self, index):
        global max_len,word_2_index
        text = self.dataset.all_data[index][:max_len]
        label =self.dataset.all_label[index]
        text_idx = [word_2_index[i] for i in text]
        text_idx += [0]*(max_len-len(text_idx))
        return text,text_idx,label
    def __next__(self):
        if self.cursor>=len(self.dataset):
            raise StopIteration
        batch_idx = self.idxlist[self.cursor:self.cursor+self.dataset.batch_size]
        self.cursor += self.dataset.batch_size
        batch_data,batch_data_idx,batch_label = [],[],[]
        for i in batch_idx:
            data,data_idx,label = self[i]
            batch_data.append(data)
            batch_data_idx.append(data_idx)
            batch_label.append(label)
        return batch_data,np.array(batch_data_idx),batch_label

def get_data():
    all_data,all_label = [],[]
    with open('data.txt','r',encoding='utf-8') as f:
        line = f.readline()
        while line:
            data,label = line.split(' ')
            all_data.append(data)
            all_label.append(int(label))
            line = f.readline()
    return all_data,all_label

def get_word_2_index(all_data):
    word_2_index = {"<PAD>":0}
    for i in all_data:
        for j in i:
            if j not in word_2_index:
                word_2_index[j] = len(word_2_index)
    return word_2_index
if __name__=="__main__":
    max_len = 20
    all_data,all_label = get_data()
    # print(all_data,all_label)
    word_2_index = get_word_2_index(all_data)
    batch_size = 2
    epoch = 10
    dataset = DataSet(all_data,all_label,batch_size)

    for e in range(epoch):
        print('*'*100)
        for data,data_idx,label in dataset:
            print(data,data_idx.shape,label)
