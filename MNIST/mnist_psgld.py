import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

train, test = chainer.datasets.get_mnist()
train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

class Mnist(chainer.Chain):
    def __init__(self):
        super(Mnist, self).__init__(
            l1 = L.Linear(None, 1000),
            l2 = L.Linear(None, 1000),
            l3 = L.Linear(None, 10),
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class pSGLD(chainer.optimizer.GradientMethod):
    def __init__(self, alpha=0.99, lam=1e-5, eta=0.01):
        self.alpha = alpha
        self.lam = lam
        self.eta = eta
    
    def init_state(self, param, state):
        state["V"] = np.zeros_like(param.data)
    
    def update_one_cpu(self, param, state):
        V = state["V"]
        g = param.grad / 100.0
        V = self.alpha * V + (1. - self.alpha) * g ** 2
        G = 1. / (np.sqrt(V) + self.lam)
        param.data += 0.5 * self.lr() * (G * g + self.gamma(param, G)) \
        + 1e-6 * np.sqrt(self.lr() * G) * np.random.randn(param.size).reshape(param.shape)
        
    def lr(self):
        return self.eta / (1. + self.t) ** 0.55
    
    def gamma(self, param, G):
        return np.zeros(param.shape)

model = L.Classifier(Mnist())
optimizer = pSGLD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='pSGLD')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()