import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import tuple_dataset
from chainer import training
from chainer.training import extensions
import original as O

train, test = np.load('train.npy'), np.load('test.npy')
X = np.arange(-10., 10., 0.1)
Y = np.arange(-15., 15., 0.1)
X, Y = np.meshgrid(X, Y)
ent = np.vstack((X.flatten(), Y.flatten())).T
matrix = np.arange(-500, 500).reshape(2, 500) / 500.

data = train.dot(matrix).astype(np.float32)
label = np.hstack((np.zeros(400), np.ones(400))).astype(np.int32)
train = tuple_dataset.TupleDataset(data, label)

data = test.dot(matrix).astype(np.float32)
label = np.hstack((np.zeros(200), np.ones(200))).astype(np.int32)
test = tuple_dataset.TupleDataset(data, label)

ent = ent.dot(matrix).astype(np.float32)

np.save('bayesian.npy', np.zeros(len(test) * 2).reshape(len(test), 2))
np.save('entropy.npy', np.zeros(60000 * 2).reshape(60000, 2))
with open('accuracy.csv', 'w'):
    pass
label = []
for i in xrange(len(test)):
    label.append(test[i][-1])
label = np.array(label).astype(np.int32)

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
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (1000, 'epoch'), out='SGD')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(O.Validation(test_iter, model, ent))
trainer.extend(O.BysAccuracy(label))

trainer.run()

p = np.load('bayesian.npy').astype(np.float32)
p /= p.sum(axis=1, keepdims=True)
y = p.argmax(axis=1)
p[p < 1e-10] = 1e-10
entropy = -np.sum(p * np.log(p), axis=1)
np.savetxt('SGDdata.csv', np.vstack([entropy, y, label]).T, delimiter=',')

p = np.load('entropy.npy').astype(np.float32)
p /= p.sum(axis=1, keepdims=True)
y = p.argmax(axis=1)
p[p < 1e-10] = 1e-10
entropy = -np.sum(p * np.log(p), axis=1)
np.save('entropy.npy', entropy)