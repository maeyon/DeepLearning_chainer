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
        super(Mnist, self).__init__(
            l1 = L.Linear(None, 500),
            l2 = L.Linear(None, 500),
            l3 = L.Linear(None, 10),
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class SGNHT(chainer.optimizer.GradientMethod):
    def __init__(self, h=0.001, A=1e-3, wei=1e-5):
        self.h = h
        self.A = A
        self.wei = wei
    
    def init_state(self, param, state):
        state["p"] = np.zeros_like(param.data)
        state["xi"] = self.A
    
    def update_one_cpu(self, param, state):
        p = state["p"]
        xi = state["xi"]
        p -= xi * p * self.h + (param.grad + self.wei * param.data) * self.h \
        + np.sqrt(2 * self.A * self.h) * np.random.randn(param.size).reshape(param.shape)
        param.data += p * self.h
        xi += (np.sum(p * p) / len(p) - 1) * self.h

model = O.Classifier(Mnist())
optimizer = SGNHT()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (30, 'epoch'), out='SGNHT')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(O.BysAccuracy(label))

trainer.run()

p = np.load('bayesian.npy').astype(np.float32)
p /= p.sum(axis=1, keepdims=True)
y = p.argmax(axis=1)
p[p < 1e-30] = 1e-30
entropy = -np.sum(p * np.log(p), axis=1)
np.savetxt('SGDdata.csv', np.vstack([entropy, y, label]).T, delimiter=',')