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

class SGNHT(chainer.optimizer.GradientMethod):
    def __init__(self, h=0.0001, A=0):
        self.h = h
        self.A = A
    
    def init_state(self, param, state):
        state["p"] = np.random.randn(param.size)
        state["xi"] = self.A
    
    def update_one_cpu(self, param, state):
        p = state["p"]
        xi = state["xi"]
        p -= xi * p * self.h + param.grad.flatten() * self.h \
        + np.sqrt(2 * self.A) * np.random.normal(0, self.h, param.size)
        param.data += p.reshape(param.shape) * self.h
        xi += (p.dot(p) / len(p) - 1) * self.h

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
		
for i in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    print "h =", i
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)
    
    model = L.Classifier(Cifar())
    optimizer = SGNHT(h=i)
    optimizer.setup(model)
    
    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (3, 'epoch'), 'SGNHT')
    
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport(log_name="log_{}".format(i)))
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()
    
#model = L.Classifier(Cifar())
#optimizer = SGNHT()
#optimizer.setup(model)

#updater = training.StandardUpdater(train_iter, optimizer)
#trainer = training.Trainer(updater, (3, 'epoch'), 'SGNHT')

#trainer.extend(extensions.Evaluator(test_iter, model))
#trainer.extend(extensions.LogReport())
#trainer.extend(extensions.PrintReport(
#['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.ProgressBar())

#trainer.run()