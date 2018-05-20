import numpy as np

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


x = X
y = Y

# randomly initialize weights
w1 = W1
w2 = W2

y_size = np.float32(M * N_O)
for t in range(EPOCHS):
    # forward pass
    h = x.dot(w1) # h = x * w1
    h_r = np.maximum(h, 0) # h_r = ReLU(h)
    y_p = h_r.dot(w2)

    # compute mean squared error and print loss
    loss = np.square(y_p - y).sum() / y_size
    print(loss)

    # backward pass: compute gradients of loss with respect to w2
    grad_y_p = 2.0 * (y_p - y) / y_size
    grad_w2 = h_r.T.dot(grad_y_p)

    # backward pass: compute gradients of loss with respect to w1
    grad_h_r = grad_y_p.dot(w2.T)
    grad_h = grad_h_r
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # update weights
    w1 -= LEARNING_RATE * grad_w1
    w2 -= LEARNING_RATE * grad_w2
