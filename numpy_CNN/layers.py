import numpy as np
from functools import reduce
import math


def im2col(image, ksize, stride):# 将张量转化为矩阵，以矩阵乘法代替卷积操作
    # 接受三维张量[W,H,c]
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col


class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1):
        self.input_shape = shape    #接收四维张量：[B,W,H,C]
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize

        # 初始化权重
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal((ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        self.delta = np.zeros((shape[0], (shape[1] - ksize + 1) // self.stride, (shape[1] - ksize + 1) // self.stride, self.output_channels))

        # 初始化梯度和输出形状
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.delta.shape

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        self.col_image = []
        conv_out = np.zeros(self.delta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.delta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def gradient(self, delta):
        self.delta = delta
        col_delta = np.reshape(delta, [self.batchsize, -1, self.output_channels])

        # 计算权重更新量
        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_delta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_delta, axis=(0, 1))

        # 将上一层的delta与反转后的卷积核进行卷积操作计算出这一层的delta
        pad_delta = np.pad(self.delta, ((0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)), 'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_delta = np.array([im2col(pad_delta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_delta = np.dot(col_pad_delta, col_flip_weights)
        next_delta = np.reshape(next_delta, self.input_shape)
        return next_delta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # L
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)


class FullyConnect(object):
    def __init__(self, shape, output_num=2):
        self.input_shape = shape
        self.batchsize = shape[0]

        input_len = reduce(lambda x, y: x * y, shape[1:])

        self.weights = np.random.standard_normal((input_len, output_num))
        self.bias = np.random.standard_normal(output_num)

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, delta):
        for i in range(delta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            delta_i = delta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, delta_i)
            self.b_gradient += delta_i.reshape(self.bias.shape)

        next_delta = np.dot(delta, self.weights.T)
        next_delta = np.reshape(next_delta, self.input_shape)

        return next_delta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        # update weights
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # set zero gradients
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)



class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0], shape[1] // self.stride, shape[2] // self.stride, self.output_channels]

    def forward(self, x):
        out = np.zeros([x.shape[0], x.shape[1] // self.stride, x.shape[2] // self.stride, self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, i // self.stride, j // self.stride, c] = np.max(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+index//self.stride, j + index % self.stride, c] = 1
        return out

    def gradient(self, delta):# 将上一层的delta扩充成池化输入张量的形状，再将没有贡献的输入元素的delta置零
        return np.repeat(np.repeat(delta, self.stride, axis=1), self.stride, axis=2) * self.index



class SoftmaxWithLoss(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.delta = np.zeros(shape)
        self.batchsize = shape[0]

    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(self.batchsize):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

        return self.loss

    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax

    def gradient(self):
        self.delta = self.softmax.copy()
        for i in range(self.batchsize):
            self.delta[i, self.label[i]] -= 1
        return self.delta

class Relu(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta