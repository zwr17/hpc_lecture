import numpy as np
import chainer.functions as F
from chainer import Variable

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


# create random input and output data
x = Variable(X)
y = Variable(Y)

# randomly initialize weights
w1 = Variable(W1)
w2 = Variable(W2)

for t in range(EPOCHS):
    # forward pass: compute predicted y
    h = F.matmul(x, w1)
    h_r = F.relu(h)
    y_p = F.matmul(h_r, w2)

    # compute and print loss
    loss = F.mean_squared_error(y_p, y)
    print(loss.data)

    # manually zero the gradients
    w1.zerograd()
    w2.zerograd()

    # backward pass
    # loss.grad = np.ones(loss.shape, dtype=np.float32)
    loss.backward()

    # update weights
    w1.data -= LEARNING_RATE * w1.grad
    w2.data -= LEARNING_RATE * w2.grad
