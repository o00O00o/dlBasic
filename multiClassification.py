import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def multiClassification(Weight1, Weight2, Label, hiddenNeuron, Samples, lr, epochs):

    weightDim = Samples.shape[1] * Samples.shape[2]
    sampleNum = Samples.shape[0]
    classNum = Weight2.shape[0]

    for epoch in range(epochs):
        for i in range(sampleNum):
            Data = Samples[i, :, :].reshape(weightDim, 1)
            label = Label[i, :].reshape(classNum, 1)

            # forward
            v1 = np.matmul(Weight1, Data)
            y1 = sigmoid(v1)
            v2 = np.matmul(Weight2, y1)
            y2 = softmax(v2)

            # backward
            e2 = label - y2
            delta2 = e2
            e1 = np.matmul(np.transpose(Weight2), delta2)
            delta1 = sigmoid(v1) * (1 - sigmoid(v1)) * e1
            dWeight1 = lr * delta1.dot(np.transpose(Data))
            Weight1 += dWeight1
            dWeight2 = lr * delta2 * np.transpose(y1)
            Weight2 += dWeight2
    return Weight1, Weight2


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

    Weight1 = np.random.rand(50, 25) * 2 - 1
    Weight2 = np.random.rand(5, 50) * 2 - 1

    Weight1, Weight2 = multiClassification(Weight1, Weight2, Label, 50, train_data, lr, epoch)

    for i in range(5):
        Data = test_data[i, :, :].reshape(25, 1)
        v1 = np.matmul(Weight1, Data)
        y1 = sigmoid(v1)
        v2 = np.matmul(Weight2, y1)
        y2 = softmax(v2)
        print("输出结果为：")
        print(y2)
        print("预测结果为：")
        print(y2.argmax() + 1)
        print("-----------")
