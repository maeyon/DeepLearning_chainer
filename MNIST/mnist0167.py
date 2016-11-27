import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

train, test = chainer.datasets.get_mnist()

train_data = []
train_target = []
for i in xrange(len(train)):
    if train[i][1] in (0, 1, 6, 7):
        train_data.append(train[i][0])
        
        if train[i][1] in (0, 1):
            train_target.append(train[i][1])
        else:
            train_target.append(train[i][1] - np.array(4, np.int32))

train = chainer.datasets.TupleDataset(train_data, train_target)
train_iter = chainer.iterators.SerialIterator(train, 100)

test_data = []
test_target = []
for i in xrange(len(test)):
    if test[i][1] in (0, 1, 6, 7):
        test_data.append(test[i][0])
        
        if test[i][1] in (0, 1):
            test_target.append(test[i][1])
        else:
            test_target.append(test[i][1] - np.array(4, np.int32))

test = chainer.datasets.TupleDataset(test_data, test_target)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

class Mnist(chainer.Chain):
    def __init__(self):
        super(Mnist, self).__init__(
            l1 = L.Linear(None, 500),
            l2 = L.Linear(None, 500),
            l3 = L.Linear(None, 4),
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = L.Classifier(Mnist())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'))

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()