import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import training, cuda
from chainer.training import extensions
import original as O

xp = cuda.cupy

train, test = xp.load('train.npy'), xp.load('test.npy')
X = np.arange(-10., 10., 0.2)
Y = np.arange(-15., 15., 0.2)
X, Y = np.meshgrid(X, Y)
ent = np.vstack((X.flatten(), Y.flatten())).T
ent = cuda.to_gpu(ent, device=0)
matrix = xp.arange(-500, 500).reshape(2, 500) / 500.

data = train.dot(matrix).astype(xp.float32)
label = xp.hstack((xp.zeros(400), xp.ones(400))).astype(xp.int32)
train = tuple_dataset.TupleDataset(data, label)

data = test.dot(matrix).astype(xp.float32)
label = xp.hstack((xp.zeros(200), xp.ones(200))).astype(xp.int32)
test = tuple_dataset.TupleDataset(data, label)

ent = ent.dot(matrix).astype(xp.float32)

xp.save('bayesian.npy', xp.zeros(len(test) * 2).reshape(len(test), 2))
xp.save('entropy.npy', xp.zeros(15000))
with open('accuracy.csv', 'w'):
    pass
label = []
for i in xrange(len(test)):
    label.append(test[i][-1])
label = np.array(label).astype(np.int32)
label = cuda.to_gpu(label, device=0)

train_iter = chainer.iterators.SerialIterator(train, 10)
test_iter = chainer.iterators.SerialIterator(test, len(test), repeat=False, shuffle=False)

class SA(chainer.Chain):
    def __init__(self):
        super(SA, self).__init__(l1 = L.Linear(None, 100),
                                 l2 = L.Linear(None, 50),
                                 l3 = L.Linear(None, 20),
                                 l4 = L.Linear(None, 2))
    
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
        param.data -= 0.5 * self.lr * (80. * g + self.weight * param.data) + np.random.normal(0, self.lr, g.shape)
        
    def update_one_gpu(self, param, state):
        gauss = xp.random.normal(0, self.lr, param.shape).astype(xp.float32)
        cuda.elementwise(
            'T grad, T lr, T weight, T gauss',
            'T param',
            'param -= 0.5 * lr * (80 * grad + weight * param) + gauss;',
            'sgld')(param.grad, self.lr, self.weight, gauss, param.data)

model = O.Classifier(SA())
cuda.get_device(0).use()
model.to_gpu()
optimizer = SGLD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (1000, 'epoch'), out='SGLD')

trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(O.Validation(test_iter, model, ent))
trainer.extend(O.BysAccuracy(label))

trainer.run()

p = xp.load('bayesian.npy').astype(xp.float32)
y = p.argmax(axis=1)
xp.save('SGLDdata.npy', xp.vstack([y, label]).T)