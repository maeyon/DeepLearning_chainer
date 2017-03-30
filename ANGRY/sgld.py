import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions
import original as O

train, test = np.load('train.npy'), np.load('test.npy')
X = np.arange(-4., 4., 0.05)
Y = np.arange(-4., 4., 0.05)
X, Y = np.meshgrid(X, Y)
ent = np.vstack((X.flatten(), Y.flatten())).T
matrix = np.arange(-500, 500).reshape(2, 500) / 500.

np.save('bayesian.npy', np.zeros(len(test) * 4).reshape(len(test), 4))
np.save('entropy.npy', np.zeros(25600))
with open('accuracy.csv', 'w'):
    pass
label = []
for i in xrange(len(test)):
    label.append(test[i][-1])
label = np.array(label).astype(np.int32)

data = train.dot(matrix).astype(np.float32)
label = np.hstack((np.zeros(200), np.ones(200), 2*np.ones(200), 3*np.ones(200))).astype(np.int32)
train = tuple_dataset.TupleDataset(data, label)

data = test.dot(matrix).astype(np.float32)
label = np.hstack((np.zeros(100), np.ones(100), 2*np.ones(100), 3*np.ones(100))).astype(np.int32)
test = tuple_dataset.TupleDataset(data, label)

ent = ent.dot(matrix).astype(np.float32)

train_iter = chainer.iterators.SerialIterator(train, 20)
test_iter = chainer.iterators.SerialIterator(test, len(test), repeat=False, shuffle=False)

class ANG(chainer.Chain):
    def __init__(self):
        super(ANG, self).__init__(l1 = L.Linear(None, 100),
                                 l2 = L.Linear(None, 100),
                                 l3 = L.Linear(None, 50),
                                 l4 = L.Linear(None, 4))
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return self.l4(h3)

class SGLD(chainer.optimizer.GradientMethod):
    def __init__(self, eta=0.001, eps=1e-4, weight=1e-3):
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
        param.data -= 0.5 * self.lr * (40 * g + self.weight * param.data) + np.random.normal(0, self.lr, g.shape)

model = O.Classifier(ANG())
optimizer = SGLD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (100, 'epoch'), out='SGLD')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(O.Validation(test_iter, model, ent))
trainer.extend(O.BysAccuracy(label))

trainer.run()

p = np.load('bayesian.npy').astype(np.float32)
y = p.argmax(axis=1)
np.save('SGLDdata.npy', np.vstack([y, label]).T)