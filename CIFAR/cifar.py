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

for i in range(1, 6):
	d = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
	train_data.extend(d["data"])
	train_target.extend(d["labels"])

d = unpickle("cifar-10-batches-py/test_batch")
test_data = d["data"]
test_target = d["labels"]

train_data = np.array(train_data).astype(np.float32).reshape((len(train_data), 3, 32, 32))
train_target = np.array(train_target).astype(np.int32)
test_data = np.array(test_data).astype(np.float32).reshape((len(test_data), 3, 32, 32))
test_target = np.array(test_target).astype(np.int32)

train_data /= 255.0
test_data /= 255.0

train = chainer.datasets.TupleDataset(train_data, train_target)
test = chainer.datasets.TupleDataset(test_data, test_target)

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

class Cifar10(chainer.Chain):
	def __init__(self):
		super(Cifar10, self).__init__(
		    conv1 = L.Convolution2D(3, 32, 3, pad=1),
		    conv2 = L.Convolution2D(32, 32, 3, pad=1),
		    l1 = L.Linear(None, 1024),
		    l2 = L.Linear(None, 10))
	
	def __call__(self, x):
		h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
		h = F.dropout(F.relu(self.l1(h)))
		return self.l2(h)
		

model = L.Classifier(Cifar10())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (10, 'epoch'))

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()