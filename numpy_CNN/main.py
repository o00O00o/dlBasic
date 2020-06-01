from layers import *
import numpy as np
import scipy.io as scio


train_num = 60000
test_num = 10000

data = scio.loadmat("dlBasic/numpy_CNN/MNISTData.mat")

images = data['X_Train'][:,:,:train_num].transpose(2, 0, 1)   # (60000, 28, 28)
labels = data['D_Train'][:, :train_num]  # (10, 60000)
labels = np.argmax(labels, 0) + 1  # (60000, )
labels[labels==10]=0

test_images = data['X_Test'][:,:,:test_num].transpose(2, 0, 1)    #(10000, 28, 28)
test_labels = data['D_Test'][:, :test_num]    #(10, 10000)
test_labels = np.argmax(test_labels, 0) + 1   #(10000, )
test_labels[test_labels==10] = 0

batch_size = 64

conv1 = Conv2D([batch_size, 28, 28, 1], 16, 5, 1)
relu1 = Relu(conv1.output_shape)
pool1 = MaxPooling(relu1.output_shape)
conv2 = Conv2D(pool1.output_shape, 32, 3, 1)
relu2 = Relu(conv2.output_shape)
pool2 = MaxPooling(relu2.output_shape)
fc = FullyConnect(pool2.output_shape, 10)
sl = SoftmaxWithLoss(fc.output_shape)


for epoch in range(20):
    if epoch < 5:
        learning_rate = 0.00001
    elif epoch < 10:
        learning_rate = 0.000001
    else:
        learning_rate = 0.0000001

    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    for i in range(images.shape[0] // batch_size):
        img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        batch_loss += sl.cal_loss(fc_out, np.array(label))
        train_loss += sl.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sl.softmax[j]) == label[j]:
                batch_acc += 1
                train_acc += 1

        conv1.gradient(relu1.gradient(pool1.gradient(conv2.gradient(relu2.gradient(pool2.gradient(fc.gradient(sl.gradient())))))))

        fc.backward(alpha=learning_rate, weight_decay=0.0004)
        conv2.backward(alpha=learning_rate, weight_decay=0.0004)
        conv1.backward(alpha=learning_rate, weight_decay=0.0004)

        if i % 10 == 0:
            print ("epoch: %d ,  batch: %3d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch, i, batch_acc / float(batch_size), batch_loss / batch_size, learning_rate))

        batch_loss = 0
        batch_acc = 0

    print ("epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (epoch, train_acc / float(train_num), train_loss / images.shape[0]))
    
    # validation
    for i in range(test_num // batch_size):
        img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
        label = test_labels[i * batch_size:(i + 1) * batch_size]
        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        val_loss += sl.cal_loss(fc_out, np.array(label))

        for j in range(batch_size):
            if np.argmax(sl.softmax[j]) == label[j]:
                val_acc += 1

    print ("epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (epoch, val_acc / float(test_num), val_loss / test_num))

