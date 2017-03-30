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

model = O.Classifier(SA())
cuda.get_device(0).use()
model.to_gpu()
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (1000, 'epoch'), out='SGD')

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
xp.save('SGDdata.npy', xp.vstack([y, label]).T)