import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import original as O

train, test = chainer.datasets.get_mnist()

np.save('bayesian.npy', np.zeros(len(test) * 10).reshape(len(test), 10))
with open('accuracy.csv', 'w'):
    pass
label = []
for i in xrange(len(test)):
    label.append(test[i][-1])
label = np.array(label).astype(np.int32)

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, len(test), repeat=False, shuffle=False)

class Mnist(chainer.Chain):
    def __init__(self):
        super(Mnist, self).__init__(l1 = L.Linear(None, 1000),
                                    l2 = L.Linear(None, 1000),
                                    l3 = L.Linear(None, 10))
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class SGLD(chainer.optimizer.GradientMethod):
    def __init__(self, eta=0.01, eps=1e-4, weight=1e-3):
        self.eta = eta
        self.eps = eps
        self.weight = weight
    
    @property
    def lr(self):
        lr = self.eta / np.power(1. + self.t, 0.55)
        if lr > self.eps:
        	return lr
        else:
        	return self.eps
    
    def update_one_cpu(self, param, state):
        g = param.grad
        param.data -= 0.5 * self.lr * (6. * g + self.weight * param.data) + np.random.normal(0, self.lr, g.shape)

model = O.Classifier(Mnist())
optimizer = SGLD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (30, 'epoch'), out='bysSGLD')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(log_name='bayesian'))
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(O.Validation(test_iter, model))
trainer.extend(O.BysAccuracy(label))

trainer.run()

p = np.load('bayesian.npy').astype(np.float32)
p /= p.sum(axis=1, keepdims=True)
y = p.argmax(axis=1)
p[p < 1e-30] = 1e-30
entropy = -np.sum(p * np.log(p), axis=1)
np.savetxt('bysSGLDdata.csv', np.vstack([entropy, y, label]).T, delimiter=',')