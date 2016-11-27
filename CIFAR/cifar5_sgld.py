import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

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
#train_iter = chainer.iterators.SerialIterator(train, 100)

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
#test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

class SGLD(chainer.optimizer.GradientMethod):
    def __init__(self, eta=0.8, eps=1e-4, delta=1e-6):
        self.eta = eta
        self.eps = eps
        self.delta = delta
    
    def lr(self):
        lr = self.eta / np.power(1.0 + self.t, 0.55)
        if lr > self.eps:
        	return lr
        else:
        	return self.eps
    
    def update_one_cpu(self, param, state):
        g = param.grad
        param.data -= 0.5 * self.lr() * g + self.delta * np.random.normal(0, self.lr(), g.shape).astype(g.dtype)

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

for i in [0.8, 0.9]:
    print 'eta =', i
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)
    model = L.Classifier(Cifar())
    optimizer = SGLD(eta=i)
    optimizer.setup(model)
    
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (20, 'epoch'), 'newSGLD')
    
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport(log_name='log_{}'.format(i)))
    trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
    
#model = L.Classifier(Cifar())
#optimizer = SGLD()
#optimizer.setup(model)

#updater = training.StandardUpdater(train_iter, optimizer)
#trainer = training.Trainer(updater, (5, 'epoch'), 'SGLD')

#trainer.extend(extensions.Evaluator(test_iter, model))
#trainer.extend(extensions.LogReport(log_name='log_com'))
#trainer.extend(extensions.PrintReport(
#['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.ProgressBar())

#trainer.run()