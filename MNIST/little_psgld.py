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

data = np.loadtxt('pSGLD/pSGLDdata.csv', delimiter=',')
index = np.argsort(data[:, 0])
image = []
for i in index[len(test)-100:]:
    image.append(test[i][0])
image = np.array(image).astype(np.float32)
train = chainer.datasets.tuple_dataset.TupleDataset(image, label[index[:100]])

train_iter = chainer.iterators.SerialIterator(train, 5)
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

class pSGLD(chainer.optimizer.GradientMethod):
    def __init__(self, alpha=0.99, lam=1e-3, eta=0.01, wei=0.01, eps=1e-4):
        self.alpha = alpha
        self.lam = lam
        self.eta = eta
        self.wei = wei
        self.eps = eps
    
    def init_state(self, param, state):
        state["v"] = np.zeros_like(param.data)
    
    def update_one_cpu(self, param, state):
        v = state["v"]
        g = param.grad
        v = self.alpha * v + (1. - self.alpha) * g ** 2
        G = 1. / (np.sqrt(v) + self.lam)
        param.data -= 0.5 * self.lr() * (G * (g + self.wei * param.data) + g) \
        + np.sqrt(self.lr() * G) * np.random.randn(param.size).reshape(param.shape)
        
    def lr(self):
        lr = self.eta / (1. + self.t) ** 0.55
        if lr > self.eps:
            return lr
        else:
            return self.eps

model = O.Classifier(Mnist())
optimizer = pSGLD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (50, 'epoch'), out='LP')

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
np.savetxt('LPdata.csv', np.vstack([entropy, y, label]).T, delimiter=',')