import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SGD(Weights, Samples, lr, epoch):
    weightDim = Weights.shape[1]
    sampleNum = Samples.shape[0]
    lossList = []
    for i in range(epoch):
        epochLoss = []
        for i in range(sampleNum):
            Data = Samples[i, :weightDim].reshape(weightDim, 1)
            label = Samples[i, -1]
            v = np.matmul(Weights, Data)
            v = v.reshape(-1)
            y = sigmoid(v)
            e = label - y
            epochLoss.append(np.square(e))
            delta = sigmoid(v) * (1 - sigmoid(v)) * e
            dW = lr * delta * Data
            Weights += dW.reshape(1, weightDim)
        lossList.append(sum(epochLoss))
    return Weights, lossList


def BGD(Weights, Samples, lr, epoch):
    weightDim = Weights.shape[1]
    sampleNum = Samples.shape[0]
    dWList = np.zeros((sampleNum, weightDim))
    lossList = []
    for i in range(epoch):
        epochLoss = []
        for i in range(sampleNum):
            Data = Samples[i, :weightDim].reshape(weightDim, 1)
            label = Samples[i, -1]
            v = np.matmul(Weights, Data)
            v = v.reshape(-1)
            y = sigmoid(v)
            e = label - y
            epochLoss.append(np.square(e))
            delta = sigmoid(v) * (1 - sigmoid(v)) * e
            dW = lr * delta * Data
            dW = dW.reshape(1, weightDim)
            dWList[i, :] = dW
        Weights += np.sum(dWList, 0) / 4
        lossList.append(sum(epochLoss))
    return Weights, lossList


def MBGD(Weights, Samples, lr, epoch, batchSize):
    weightDim = Weights.shape[1]
    sampleNum = Samples.shape[0]
    steps = int(sampleNum / batchSize)
    lossList = []
    for i in range(epoch):
        epochLoss = []
        for step in range(steps):
            dWList = np.zeros((batchSize, weightDim))
            batchData = Samples[batchSize * step:batchSize * (step + 1), :weightDim]
            batchLabel = Samples[batchSize * step:batchSize * (step + 1), -1]
            for i in range(batchSize):
                Data = batchData[i, :weightDim].reshape(weightDim, 1)
                label = batchLabel[i]
                v = np.matmul(Weights, Data)
                v = v.reshape(-1)
                y = sigmoid(v)
                e = label - y
                epochLoss.append(np.square(e))
                delta = sigmoid(v) * (1 - sigmoid(v)) * e
                dW = lr * delta * Data
                dW = dW.reshape(1, weightDim)
                dWList[i, :] = dW
            Weights += np.sum(dWList, 0) / batchSize
        lossList.append(sum(epochLoss))
    return Weights, lossList


if __name__ == "__main__":

    InitWeights = np.random.rand(1, 3) * 2 - 1
    Samples = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1]])
    lr = 0.9
    epoch = 1000

    SGD_Weights, SGD_loss = SGD(InitWeights, Samples, lr, epoch)
    BGD_Weights, BGD_loss = BGD(InitWeights, Samples, lr, epoch)
    MBGD_Weights, MBGD_loss = MBGD(InitWeights, Samples, lr, epoch, batchSize=2)

    index = range(epoch)

    plt.figure(figsize=(8, 8))
    plt.plot(index, SGD_loss)
    plt.plot(index, BGD_loss)
    plt.plot(index, MBGD_loss)
    plt.show()
