import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

train, test = chainer.datasets.get_mnist()
train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

class Mnist(chainer.Chain):
    def __init__(self):
        super(Mnist, self).__init__(
            l1 = L.Linear(None, 1000),
            l2 = L.Linear(None, 1000),
            l3 = L.Linear(None, 10),
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class SGLD(chainer.optimizer.GradientMethod):
    def __init__(self, eta=0.8, eps=1e-4, delta=1e-6):
        self.eta = eta
        self.eps = eps
        self.delta = delta
    
    def lr(self):
        lr = self.eta / np.power(1.0 + self.t, 0.55)
        if lr > self.eps:
        	return lr
        else:
        	return self.eps
    
    def update_one_cpu(self, param, state):
        g = param.grad
        param.data -= 0.5 * self.lr() * g + self.delta * np.random.normal(0, self.lr(), g.shape).astype(g.dtype)

model = L.Classifier(Mnist())
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='SGLD')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()