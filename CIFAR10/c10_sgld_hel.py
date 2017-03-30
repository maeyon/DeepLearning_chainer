import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S
from chainer import cuda
from chainer import training
from chainer.training import extensions
import coriginal as C

xp = cuda.cupy

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
	train_data.extend(d["data"])
	train_target.extend(d["labels"])

train_data = xp.array(train_data).astype(xp.float32).reshape((len(train_data), 3, 32, 32))
train_target = xp.array(train_target).astype(xp.int32)
train_data /= 255.0

train = chainer.datasets.TupleDataset(train_data, train_target)
train_iter = chainer.iterators.SerialIterator(train, 100)
train_entr_iter = chainer.iterators.SerialIterator(train, 10000, repeat=False, shuffle=False)

test_data = []
test_target = []

d = unpickle("cifar-10-batches-py/test_batch")
test_data.extend(d["data"])
test_target.extend(d["labels"])

test_data = xp.array(test_data).astype(xp.float32).reshape((len(test_data), 3, 32, 32))
test_target = xp.array(test_target).astype(xp.int32)
test_data /= 255.0

test = chainer.datasets.TupleDataset(test_data, test_target)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

xp.save('bayesian.npy', xp.zeros(len(test_data) * 10).reshape(len(test_data), 10))
xp.save('entropy.npy', xp.zeros(len(test_data)))
xp.save('entr.npy', xp.zeros(len(train_data)))
with open('accuracy.csv', 'w'):
    pass

class Cifar(chainer.Chain):
    def __init__(self):
        super(Cifar, self).__init__(
            conv1 = L.Convolution2D(3, 32, 3, pad=1),
            conv2 = L.Convolution2D(32, 32, 3, pad=1),
            l1 = L.Linear(None, 512),
            l2 = L.Linear(None, 10),
            bnorm1 = L.BatchNormalization(32),
            bnorm2 = L.BatchNormalization(32))
        
    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = self.bnorm1(h)
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = self.bnorm2(h)
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.l1(h)))
        return self.l2(h)
		
class SGLD(chainer.optimizer.GradientMethod):
    def __init__(self, eta=0.001, eps=1e-4, weight=1e-4):
        self.eta = eta
        self.eps = eps
        self.weight = weight
    
    @property
    def lr(self):
        lr = self.eta / xp.power(1. + self.t, 0.55).astype(xp.float32)
        if lr > self.eps:
        	return lr
        else:
        	return self.eps
    
    def update_one_cpu(self, param, state):
        g = param.grad
        param.data -= 0.5 * self.lr * (500 * g + self.weight * param.data) + np.random.normal(0, self.lr, g.shape)
        
    def update_one_gpu(self, param, state):
        gauss = xp.random.normal(0, self.lr, param.shape).astype(xp.float32)
        cuda.elementwise(
            'T grad, T lr, T weight, T gauss',
            'T param',
            'param -= 0.5 * lr * (500 * grad + weight * param) + gauss;',
            'sgld')(param.grad, self.lr, self.weight, gauss, param.data)

model = C.Classifier(Cifar())
cuda.get_device(0).use()
model.to_gpu()
optimizer = SGLD()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (10, 'epoch'), 'HEL')

trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(C.Validation(test_iter, model))
trainer.extend(C.BysAccuracy(test_target))
trainer.extend(C.HelTrain(train_entr_iter, model))

trainer.run()
entropy = np.load('entr.npy').astype(np.float32)

entropy_cpu = cuda.to_cpu(entropy)
index_cpu = np.argsort(entropy_cpu)
index = cuda.to_gpu(index_cpu, device=0)

S.save_npz('learned_model', model)
S.save_npz('learned_opt', optimizer)

bysac = np.loadtxt('accuracy.csv', delimiter=',')
bys = np.load('bayesian.npy')
entropy = np.load('entropy.npy')

with open('accuracy.csv', 'w'):
    pass

for i in xrange(1, 5):
    train = chainer.datasets.TupleDataset(train_data[index[-1:-10000*i-1:-1]], train_target[index[-1:-10000*i-1:-1]])
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, len(test_data), repeat=False, shuffle=False)

    S.load_npz('learned_model', model)
    S.load_npz('learned_opt', optimizer)

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (40, 'epoch'), 'HEL')

    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.LogReport(log_name='hel_{}'.format(i)))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(C.Validation(test_iter, model))
    trainer.extend(C.BysAccuracy(test_target))

    trainer.run()
    
    np.savetxt('acc_{}.csv'.format(i), np.loadtxt('accuracy.csv', delimiter=','), delimiter=',')
    np.savetxt('accuracy.csv', bysac, delimiter=',')

train = chainer.datasets.TupleDataset(train_data, train_target)
train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, len(test_data), repeat=False, shuffle=False)

S.load_npz('learned_model', model)
S.load_npz('learned_opt', optimizer)

np.save('entropy.npy', entropy)

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (40, 'epoch'), 'HEL')

trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.LogReport(log_name='no_hel'))
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
trainer.extend(C.Validation(test_iter, model))
trainer.extend(C.BysAccuracy(test_target))

trainer.run()