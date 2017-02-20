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
    
class Adam(chainer.optimizer.GradientMethod):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight=1e-4):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight = weight
        
    def init_state(self, param, state):
        state['m'] = np.zeros_like(param.data)
        state['v'] = np.zeros_like(param.data)
    
    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return self.alpha * np.sqrt(fix2) / fix1
    
    def update_one_cpu(self, param, state):
        m, v = state['m'], state['v']
        grad = 600 * param.grad + self.weight * param.data
        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        G = 1. / (self.eps + np.sqrt(v))
        param.data -= 0.5 * self.lr * G * m + np.sqrt(self.lr * G) * np.random.randn(param.size).reshape(param.shape)

model = O.Classifier(Mnist())
optimizer = Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (30, 'epoch'), out='Adam')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
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
np.savetxt('Adamdata.csv', np.vstack([entropy, y, label]).T, delimiter=',')