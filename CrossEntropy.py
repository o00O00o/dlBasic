import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def CE(d, y):
    return -d * np.log(y) - (1 - d) * np.log(1 - y)


def flexBackProp(Weight1, Weight2, hiddenNeuron, Samples, lr, epochs):

    errorList = np.zeros(epochs)

    weightDim = Samples.shape[1] - 1
    sampleNum = Samples.shape[0]

    for epoch in range(epochs):
        epochError = []
        for i in range(sampleNum):
            Data = Samples[i, :weightDim].reshape(weightDim, 1)
            label = Samples[i, -1]

            # forward
            v1 = np.matmul(Weight1, Data)
            y1 = sigmoid(v1)
            v2 = np.matmul(Weight2, y1)
            y2 = sigmoid(v2)

            # backward
            e2 = label - y2
            epochError.append(np.square(e2))
            delta2 = sigmoid(v2) * (1 - sigmoid(v2)) * e2
            e1 = np.transpose(Weight2) * delta2
            delta1 = sigmoid(v1) * (1 - sigmoid(v1)) * e1
            dWeight1 = lr * delta1.dot(np.transpose(Data))
            Weight1 += dWeight1
            dWeight2 = lr * delta2 * np.transpose(y1)
            Weight2 += dWeight2
        errorList[epoch] = sum(epochError)
    return Weight1, Weight2, errorList.reshape(epochs,)


def flexCEBackProp(Weight1, Weight2, hiddenNeuron, Samples, lr, epochs):

    errorList = np.zeros(epochs)

    weightDim = Samples.shape[1] - 1
    sampleNum = Samples.shape[0]

    for epoch in range(epochs):
        epochError = []
        for i in range(sampleNum):
            Data = Samples[i, :weightDim].reshape(weightDim, 1)
            label = Samples[i, -1]

            # forward
            v1 = np.matmul(Weight1, Data)
            y1 = sigmoid(v1)
            v2 = np.matmul(Weight2, y1)
            y2 = sigmoid(v2)

            # backward
            e2 = label - y2
            epochError.append(np.square(e2))
            delta2 = e2
            e1 = np.transpose(Weight2) * delta2
            delta1 = sigmoid(v1) * (1 - sigmoid(v1)) * e1
            dWeight1 = lr * delta1.dot(np.transpose(Data))
            Weight1 += dWeight1
            dWeight2 = lr * delta2 * np.transpose(y1)
            Weight2 += dWeight2
        errorList[epoch] = sum(epochError)
    return Weight1, Weight2, errorList.reshape(epochs,)


if __name__ == "__main__":

    Samples = np.array([[0, 0, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 0]])
    lr = 1e-3
    epoch = 10000

    Weight1 = np.random.rand(4, 3) * 2 - 1
    Weight2 = np.random.rand(1, 4) * 2 - 1

    sqWeight1, sqWeight2, sqeError = flexBackProp(Weight1, Weight2, 4, Samples, lr, epoch)
    CEWeight1, CEWeight2, CEError = flexCEBackProp(Weight1, Weight2, 4, Samples, lr, epoch)
    print(CEError[0])

    for i in range(4):
        Data = Samples[i, :3]
        v1 = np.matmul(sqWeight1, Data)
        y1 = sigmoid(v1)
        v2 = np.matmul(sqWeight2, y1)
        y2 = sigmoid(v2)
        print(y2)

    print("---------------------------------")

    for i in range(4):
        Data = Samples[i, :3]
        v1 = np.matmul(CEWeight1, Data)
        y1 = sigmoid(v1)
        v2 = np.matmul(CEWeight2, y1)
        y2 = sigmoid(v2)
        print(y2)

    index = range(epoch)

    plt.figure(figsize=(8, 8))
    plt.plot(index, sqeError, label="SQError")
    plt.plot(index, CEError, label="CEError")
    plt.legend()
    plt.show()
