import numpy as np
import chainer
import chainer.functions as F
import chainer.optimizers as P
import chainer.links as L
import chainer.datasets as D
import chainer.iterators as Iter
from chainer import training
from chainer.training import extensions
from chainer import reporter

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


class LossCalculator(chainer.Chain):
    def __init__(self, model):
        super(LossCalculator, self).__init__()
        with self.init_scope():
            self.model = model

    def __call__(self, x, y):
        y_p = self.model(x)
        loss = F.mean_squared_error(y_p, y)
        reporter.report({'loss': loss}, self)
        return loss


# make a iterator
dataset = D.TupleDataset(X, Y)
train_iter = Iter.SerialIterator(dataset, batch_size=M, shuffle=False)

# create a network
model = TwoLayerNet(N_I, N_H, N_O)
loss_calculator = LossCalculator(model)

# create an optimizer
optimizer = P.SGD(lr=LEARNING_RATE)

# connect the optimizer with the network
optimizer.setup(loss_calculator)

# make a updater
updater = training.StandardUpdater(train_iter, optimizer)

# make a trainer
trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out='result')
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))

trainer.run()
