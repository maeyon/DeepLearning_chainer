import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, cuda
from chainer.training import extensions
import original as O

xp = cuda.cupy

train, test = chainer.datasets.get_mnist()

xp.save('bayesian.npy', xp.zeros(len(test) * 10).reshape(len(test), 10))
xp.save('entropy.npy', xp.zeros(len(test)))
with open('accuracy.csv', 'w'):
    pass
label = []
for i in xrange(len(test)):
    label.append(test[i][-1])
label = xp.array(label).astype(xp.int32)

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, len(test), repeat=False, shuffle=False)

class Mnist(chainer.Chain):
    def __init__(self):
        super(Mnist, self).__init__(
            l1 = L.Linear(None, 500),
            l2 = L.Linear(None, 500),
            l3 = L.Linear(None, 10),
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = O.Classifier(Mnist())
cuda.get_device(0).use()
model.to_gpu()
optimizer = chainer.optimizers.SGD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (30, 'epoch'), out='SGD')

trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(O.Validation(test_iter, model))
trainer.extend(O.BysAccuracy(label))

trainer.run()

p = xp.load('bayesian.npy').astype(xp.float32)
y = p.argmax(axis=1)
h = xp.load('entropy.npy').astype(xp.float32)
h /= 18000. * xp.log(10)
xp.save('SGDdata.npy', xp.vstack([h, y, label]).T)