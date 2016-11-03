from __future__ import print_function
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

train, test = chainer.datasets.get_mnist()

img = []
lbl = []
for i in xrange(len(train)):
    if train[i][1] in (0, 1, 6, 7):
        img.append(train[i][0])
        
        if train[i][1] in (0, 1):
            lbl.append(train[i][1])
        
        else:
            lbl.append(train[i][1] - np.array(4, np.int32))
    

train_set = chainer.datasets.TupleDataset(img, lbl)
train_iter = chainer.iterators.SerialIterator(train_set, 100)

img = []
lbl = []
for i in xrange(len(test)):
    if test[i][1] in (0, 1, 6, 7):
        img.append(test[i][0])
        
        if test[i][1] in (0, 1):
            lbl.append(test[i][1])
        
        else:
            lbl.append(test[i][1] - np.array(4, np.int32))


test_set = chainer.datasets.TupleDataset(img, lbl)
test_iter = chainer.iterators.SerialIterator(test_set, 100, repeat=False, shuffle=False)

class MNT(chainer.Chain):
    def __init__(self):
        super(MNT, self).__init__(
            l1 = L.Linear(None, 500),
            l2 = L.Linear(None, 500),
            l3 = L.Linear(None, 4),
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
    

model = L.Classifier(MNT())

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'))

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger = (20, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()