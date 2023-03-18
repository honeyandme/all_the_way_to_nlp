import numpy as np


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    res = np.clip(res, 1e-10, 0.99999999)
    return res


if __name__ == "__main__":
    np.random.seed(100)
    dogs = np.array([[8.9, 12], [9, 11], [10, 13], [9.9, 11.2], [12.2, 10.1], [9.8, 13], [8.8, 11.2]],
                    dtype=np.float32)  # 0
    cats = np.array([[3, 4], [5, 6], [3.5, 5.5], [4.5, 5.1], [3.4, 4.1], [4.1, 5.2], [4.4, 4.4]], dtype=np.float32)
    label = np.array([0] * 7 + [1] * 7, dtype=np.int32).reshape(-1, 1)
    X = np.vstack((dogs, cats))

    Hidden_size = 30
    w1 = np.random.normal(0, 0.5, (X.shape[1], Hidden_size))
    w2 = np.random.normal(0, 0.5, (Hidden_size, 1))
    b1 = 0
    b2 = 0

    lr = 0.1
    epoch = 100000
    for e in range(epoch):
        H = X @ w1 + b1
        H_s = sigmoid(H)
        pre = H @ w2 + b2
        pre_s = sigmoid(pre)
        loss = -np.mean(label * np.log(pre_s) + (1 - label) * np.log(1 - pre_s))
        G = (pre_s - label) / X.shape[0]


        delta_w2 = H_s.T @ G
        delta_H_s = G @ w2.T
        delta_H = delta_H_s*((sigmoid(H)*(1-sigmoid(H))))
        delta_w1 = X.T @ delta_H
        delta_b2 = np.sum(G)
        delta_b1 = np.sum(delta_H)


        w1 -= lr * delta_w1
        w2 -= lr * delta_w2
        b1 -= delta_b1
        b2 -= delta_b2

        if(e%10000==0):
            print(f'loss={loss:.20f}')

    while (True):
        x1 = float(input('请输入第一个特征:'))
        x2 = float(input('请输入第二个特征:'))
        pre = np.array([[x1, x2]]) @ w1 + b1
        pre = pre @ w2+b2
        if (pre < 0):
            print('dog')
        else:
            print('cat')
