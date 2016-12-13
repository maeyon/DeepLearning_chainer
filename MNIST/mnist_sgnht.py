import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

train, test = chainer.datasets.get_mnist()
#train_iter = chainer.iterators.SerialIterator(train, 100)
#test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

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

class SGNHT(chainer.optimizer.GradientMethod):
    def __init__(self, h=0.0001, A=0):
        self.h = h
        self.A = A
    
    def init_state(self, param, state):
        state["p"] = np.random.randn(param.size)
        state["xi"] = self.A
    
    def update_one_cpu(self, param, state):
        p = state["p"]
        xi = state["xi"]
        p -= xi * p * self.h + param.grad.flatten() * self.h \
        + np.sqrt(2 * self.A) * np.random.normal(0, self.h, param.size)
        param.data += p.reshape(param.shape) * self.h
        xi += (p.dot(p) / len(p) - 1) * self.h

for i in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
    print "h =", i
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)
    model = L.Classifier(Mnist())
    optimizer = SGNHT(h=i)
    optimizer.setup(model)
    
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (5, 'epoch'), out='SGNHT')
    
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport(log_name="log_{}".format(i)))
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
#model = L.Classifier(Mnist())
#optimizer = chainer.optimizers.SGD()
#optimizer.setup(model)

#updater = training.StandardUpdater(train_iter, optimizer)
#trainer = training.Trainer(updater, (20, 'epoch'), out='SGNHT')

#trainer.extend(extensions.Evaluator(test_iter, model))
#trainer.extend(extensions.LogReport())
#trainer.extend(extensions.PrintReport(
#['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.ProgressBar())

#trainer.run()