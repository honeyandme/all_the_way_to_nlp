import numpy as np
import struct
def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)

def label2onehot(labels, class_num=10):
    num = labels.shape[0]
    one_hot_label = np.zeros((num, class_num))
    for i, x in enumerate(labels):
        one_hot_label[i][x] = 1
    return one_hot_label

def sigmoid(x):
    x = np.clip(x,-1e20,1e20)
    return 1/(1+np.exp(-x))

def softmax(x):
    x = np.exp(x)
    e_sum = np.sum(x,axis=1,keepdims=True)
    x /= e_sum
    return x