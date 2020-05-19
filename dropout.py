import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def ReLU(x):
    return np.maximum(x, 0)


def ReLuPrime(x):
    return np.where(x > 0, 1, 0)


def dropout(x, ratio):
    sample = np.random.binomial(n=1, p=1 - ratio, size=x.shape)
    x *= sample
    x /= 1 - ratio
    return x


def deepReLU(Weight1, Weight2, Weight3, Weight4, Label, Samples, lr, epochs):

    weightDim = Samples.shape[1] * Samples.shape[2]
    sampleNum = Samples.shape[0]
    classNum = Weight4.shape[0]

    for epoch in range(epochs):
        for i in range(sampleNum):
            Data = Samples[i, :, :].reshape(weightDim, 1)
            label = Label[i, :].reshape(classNum, 1)

            # forward
            v1 = np.matmul(Weight1, Data)
            y1 = ReLU(v1)
            y1 = dropout(y1, 0.2)

            v2 = np.matmul(Weight2, y1)
            y2 = ReLU(v2)
            y2 = dropout(y2, 0.2)

            v3 = np.matmul(Weight3, y2)
            y3 = ReLU(v3)
            y3 = dropout(y3, 0.2)

            v4 = np.matmul(Weight4, y3)
            y4 = softmax(v4)

            # backward
            e4 = label - y4
            delta4 = e4
            e3 = np.matmul(np.transpose(Weight4), delta4)
            delta3 = ReLuPrime(v3) * e3
            e2 = np.matmul(np.transpose(Weight3), delta3)
            delta2 = ReLuPrime(v2) * e2
            e1 = np.matmul(np.transpose(Weight2), delta2)
            delta1 = ReLuPrime(v1) * e1
            # updata the weights
            Weight1 += lr * delta1.dot(np.transpose(Data))
            Weight2 += lr * delta2 * np.transpose(y1)
            Weight3 += lr * delta3 * np.transpose(y2)
            Weight4 += lr * delta4 * np.transpose(y3)

    return Weight1, Weight2, Weight3, Weight4


if __name__ == "__main__":

    train_data = np.zeros((5, 5, 5))
    train_data[0, :, :] = np.array([[0., 1., 1., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 0., 1., 0., 0.],
                                    [0., 1., 1., 1., 0.]])
    train_data[1, :, :] = np.array([[1., 1., 1., 1., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 1., 1., 1., 0.],
                                    [1., 0., 0., 0., 0.],
                                    [1., 1., 1., 1., 1.]])
    train_data[2, :, :] = np.array([[1., 1., 1., 1., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [0., 1., 1., 1., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [1., 1., 1., 1., 0.]])
    train_data[3, :, :] = np.array([[0., 0., 0., 1., 0.],
                                    [0., 0., 1., 1., 0.],
                                    [0., 1., 0., 1., 0.],
                                    [1., 1., 1., 1., 1.],
                                    [0., 0., 0., 1., 0.]])
    train_data[4, :, :] = np.array([[1., 1., 1., 1., 1.],
                                    [1., 0., 0., 0., 0.],
                                    [1., 1., 1., 1., 0.],
                                    [0., 0., 0., 0., 1.],
                                    [1., 1., 1., 1., 0.]])

    test_data = np.zeros((5, 5, 5))
    test_data[0, :, :] = np.array([[0., 0., 1., 1., 0.],
                                   [0., 0., 1., 1., 0.],
                                   [0., 1., 0., 1., 0.],
                                   [0., 0., 0., 1., 0.],
                                   [0., 1., 1., 1., 0.]])
    test_data[1, :, :] = np.array([[1., 1., 1., 1., 0.],
                                   [0., 0., 0., 0., 1.],
                                   [0., 1., 1., 1., 0.],
                                   [1., 0., 0., 0., 1.],
                                   [1., 1., 1., 1., 1.]])
    test_data[2, :, :] = np.array([[1., 1., 1., 1., 0.],
                                   [0., 0., 0., 0., 1.],
                                   [0., 1., 1., 1., 0.],
                                   [1., 0., 0., 0., 1.],
                                   [1., 1., 1., 1., 0.]])
    test_data[3, :, :] = np.array([[0., 1., 1., 1., 0.],
                                   [0., 1., 0., 0., 0.],
                                   [0., 1., 1., 1., 0.],
                                   [1., 0., 0., 1., 1.],
                                   [0., 1., 1., 1., 0.]])
    test_data[4, :, :] = np.array([[0., 1., 1., 1., 1.],
                                   [0., 1., 0., 0., 0.],
                                   [0., 1., 1., 1., 0.],
                                   [0., 0., 0., 1., 0.],
                                   [1., 1., 1., 1., 0.]])

    Label = np.array([[1., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 1.]])

    lr = 1e-3
    epoch = 1000

    Weight1 = np.random.rand(20, 25) * 2 - 1
    Weight2 = np.random.rand(20, 20) * 2 - 1
    Weight3 = np.random.rand(20, 20) * 2 - 1
    Weight4 = np.random.rand(5, 20) * 2 - 1

    Weight1, Weight2, Weight3, Weight4 = deepReLU(Weight1, Weight2, Weight3, Weight4, Label, train_data, lr, epoch)

    for i in range(5):
        Data = test_data[i, :, :].reshape(25, 1)
        v1 = np.matmul(Weight1, Data)
        y1 = ReLU(v1)

        v2 = np.matmul(Weight2, y1)
        y2 = ReLU(v2)

        v3 = np.matmul(Weight3, y2)
        y3 = ReLU(v3)

        v4 = np.matmul(Weight4, y3)
        y4 = softmax(v4)
        print("输出结果为：")
        print(y4)
        print("预测结果为：")
        print(y4.argmax() + 1)
        print("-----------")
