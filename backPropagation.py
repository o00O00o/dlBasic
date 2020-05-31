import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def BackProp(Weight1, Weight2, Samples, lr, epoch):
    weightDim = Samples.shape[1] - 1
    sampleNum = Samples.shape[0]
    for i in range(epoch):
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
            delta2 = sigmoid(v2) * (1 - sigmoid(v2)) * e2
            e1 = np.transpose(Weight2) * delta2
            delta1 = sigmoid(v1) * (1 - sigmoid(v1)) * e1
            dWeight1 = lr * delta1.dot(np.transpose(Data))
            Weight1 += dWeight1
            dWeight2 = lr * delta2 * np.transpose(y1)
            Weight2 += dWeight2

    return Weight1, Weight2


def flexBackProp(hiddenNeuron, Samples, lr, epoch):

    weightDim = Samples.shape[1] - 1
    sampleNum = Samples.shape[0]

    # weights initialization
    Weight1 = np.random.rand(hiddenNeuron, weightDim) * 2 - 1
    Weight2 = np.random.rand(1, hiddenNeuron) * 2 - 1

    for i in range(epoch):
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
            delta2 = sigmoid(v2) * (1 - sigmoid(v2)) * e2
            e1 = np.transpose(Weight2) * delta2
            delta1 = sigmoid(v1) * (1 - sigmoid(v1)) * e1
            dWeight1 = lr * delta1.dot(np.transpose(Data))
            Weight1 += dWeight1
            dWeight2 = lr * delta2 * np.transpose(y1)
            Weight2 += dWeight2

    return Weight1, Weight2


def flexMmtBackProp(hiddenNeuron, Samples, lr, epoch, beta):

    weightDim = Samples.shape[1] - 1
    sampleNum = Samples.shape[0]

    # weights initialization
    Weight1 = np.random.rand(hiddenNeuron, weightDim) * 2 - 1
    Weight2 = np.random.rand(1, hiddenNeuron) * 2 - 1

    # moments initialization
    Mmt1 = np.zeros((hiddenNeuron, weightDim))
    Mmt2 = np.zeros((1, hiddenNeuron))

    for i in range(epoch):
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
            delta2 = sigmoid(v2) * (1 - sigmoid(v2)) * e2
            e1 = np.transpose(Weight2) * delta2
            delta1 = sigmoid(v1) * (1 - sigmoid(v1)) * e1
            dWeight1 = lr * delta1.dot(np.transpose(Data))
            Mmt1 = dWeight1 + beta * Mmt1
            Weight1 += Mmt1
            dWeight2 = lr * delta2 * np.transpose(y1)
            Mmt2 = dWeight2 + beta * Mmt2
            Weight2 += Mmt2

    return Weight1, Weight2


if __name__ == "__main__":

    Samples = np.array([[0, 0, 1, 0], [0, 1, 1, 1],
                        [1, 0, 1, 1], [1, 1, 1, 0]])
    lr = 0.9
    epoch = 1000
    beta = 0.9

    Weight1, Weight2 = flexMmtBackProp(4, Samples, lr, epoch, beta)

    for i in range(4):
        Data = Samples[i, :3]
        v1 = np.matmul(Weight1, Data)
        y1 = sigmoid(v1)
        v2 = np.matmul(Weight2, y1)
        y2 = sigmoid(v2)
        print(y2)
