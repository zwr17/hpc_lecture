import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

EPOCHS = 300
M = 64
N_I = 1000
N_H = 100
N_O = 10
LEARNING_RATE = 1.0e-04

# set a specified seed to random value generator in order to reproduce the same results
np.random.seed(1)

X = np.random.randn(M, N_I).astype(np.float32)
Y = np.random.randn(M, N_O).astype(np.float32)
W1 = np.random.randn(N_I, N_H).astype(np.float32)
W2 = np.random.randn(N_H, N_O).astype(np.float32)


class TwoLayerNet(chainer.Chain):
    def __init__(self, d_in, h, d_out):
        super(TwoLayerNet, self).__init__(
            linear1=L.Linear(d_in, h,  initialW=W1.transpose()),
            linear2=L.Linear(h, d_out, initialW=W2.transpose())
        )

    def __call__(self, x):
        g = self.linear1(x)
        h_r = F.relu(g)
        y_p = self.linear2(h_r)
        return y_p

# create random input and output data
x = Variable(X)
y = Variable(Y)

# create a network
model = TwoLayerNet(N_I, N_H, N_O)

for t in range(EPOCHS):
    # forward
    y_p = model(x)

    # compute and print loss
    loss = F.mean_squared_error(y_p, y)
    print(loss.data)

    # zero the gradients
    model.cleargrads()

    # backward
    loss.backward()

    # update weights
    model.linear1.W.data -= LEARNING_RATE * model.linear1.W.grad
    model.linear2.W.data -= LEARNING_RATE * model.linear2.W.grad
