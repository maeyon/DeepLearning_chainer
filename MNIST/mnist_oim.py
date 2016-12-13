import numpy as np
from scipy import sparse, io
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
            l1 = L.Linear(784, 1000),
            l2 = L.Linear(1000, 1000),
            l3 = L.Linear(1000, 10),
        )
    
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class OIm(chainer.optimizer.GradientMethod):
    def __init__(self, gamma=2., eta=0.01, eps=1e-4, T=1e-6):
        self.gamma = gamma
        self.eta = eta
        self.eps = eps
        self.T = T
    
    def update_one_cpu(self, param, state):
        n = param.size
        g = param.grad.flatten()
        M = io.loadmat('MNIST_{}'.format(int(self.gamma)))['m_{}'.format(n)]
        param.data += self.lr() * (M.dot(g) + self.T * np.random.randn(n)).reshape(param.shape)
        
    def lr(self):
        lr = self.eta / (1. + self.t) ** 0.55
        if lr < self.eps:
            return self.eps
        else:
            return lr

gamma = 2.
model = L.Classifier(Mnist())
optimizer = OIm(gamma)
optimizer.setup(model)

d = dict()

for param in optimizer.target.params():
    n = param.size
    M = sparse.lil_matrix((n, n), dtype=param.dtype)
    M[0, :2] = [-1., -gamma]
    M[0, n-1] = gamma
    for i in xrange(1, n-1):
        M[i, i-1:i+2] = [gamma, -1., -gamma]
    M[n-1, n-2:] = [gamma, -1.]
    M[n-1, 0] = -gamma
    d['m_{}'.format(n)] = M
    print 'save', n
io.savemat('MNIST_{}'.format(int(gamma)), d)
print 'save matrix'

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='OImethod')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(log_name='log_{}'.format(int(gamma))))
trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()