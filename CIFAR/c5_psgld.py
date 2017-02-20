import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import original as O

def unpickle(file):
	import cPickle
	fo = open(file, "rb")
	dict = cPickle.load(fo)
	fo.close()
	return dict

train_data = []
train_target = []

for i in xrange(1, 6):
	d = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
	for j in xrange(len(d["data"])):
		if d["labels"][j] in xrange(5):
			train_data.append(d["data"][j])
			train_target.append(d["labels"][j])

train_data = np.array(train_data).astype(np.float32).reshape((len(train_data), 3, 32, 32))
train_target = np.array(train_target).astype(np.int32)
train_data /= 255.0

train = chainer.datasets.TupleDataset(train_data, train_target)
train_iter = chainer.iterators.SerialIterator(train, 100)

test_data = []
test_target = []

d = unpickle("cifar-10-batches-py/test_batch")
for j in xrange(len(d["data"])):
	if d["labels"][j] in xrange(5):
		test_data.append(d["data"][j])
		test_target.append(d["labels"][j])

test_data = np.array(test_data).astype(np.float32).reshape((len(test_data), 3, 32, 32))
test_target = np.array(test_target).astype(np.int32)
test_data /= 255.0

test = chainer.datasets.TupleDataset(test_data, test_target)
test_iter = chainer.iterators.SerialIterator(test, len(test_data), repeat=False, shuffle=False)

np.save('bayesian.npy', np.zeros(len(test_data) * 5).reshape(len(test_data), 5))
with open('accuracy.csv', 'w'):
    pass


class Cifar(chainer.Chain):
	def __init__(self):
		super(Cifar, self).__init__(
		    conv1 = L.Convolution2D(3, 32, 3, pad=1),
		    conv2 = L.Convolution2D(32, 32, 3, pad=1),
		    l1 = L.Linear(None, 512),
		    l2 = L.Linear(None, 5))
	
	def __call__(self, x):
		h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
		h = F.dropout(F.relu(self.l1(h)))
		return self.l2(h)
		
class pSGLD(chainer.optimizer.GradientMethod):
    def __init__(self, alpha=0.999, lam=1e-4, eta=0.001, wei=1e-4, eps=1e-4):
        self.alpha = alpha
        self.lam = lam
        self.eta = eta
        self.wei = wei
        self.eps = eps
    
    def init_state(self, param, state):
        state["v"] = np.zeros_like(param.data)
    
    def update_one_cpu(self, param, state):
        v = state["v"]
        g = param.grad / 100.0
        v = self.alpha * v + (1. - self.alpha) * g ** 2
        G = 1. / (np.sqrt(v) + self.lam)
        param.data -= 0.5 * self.lr() * (G * (50000 * g + self.wei * param.data)) \
        + np.sqrt(self.lr() * G) * np.random.randn(param.size).reshape(param.shape)
        
    def lr(self):
        lr = self.eta / (1. + self.t) ** 0.55
        if lr > self.eps:
            return lr
        else:
            return self.eps

model = O.Classifier(Cifar())
optimizer = pSGLD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (2, 'epoch'), 'pSGLD')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
#trainer.extend(O.Validation(test_iter, model))
#trainer.extend(O.BysAccuracy(test_target))

trainer.run()

p = np.load('bayesian.npy').astype(np.float32)
p /= p.sum(axis=1, keepdims=True)
y = p.argmax(axis=1)
p[p < 1e-30] = 1e-30
entropy = -np.sum(p * np.log(p), axis=1)
np.savetxt('pSGLDdata.csv', np.vstack([entropy, y, test_target]).T, delimiter=',')