import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers as S
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

#train_entr_iter = chainer.iterators.SerialIterator(train, len(train), repeat=False, shuffle=False)

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

#np.save('bayesian.npy', np.zeros(len(train_data) * 5).reshape(len(train_data), 5))
#with open('accuracy.csv', 'w'):
#    pass

class Cifar(chainer.Chain):
    def __init__(self):
        super(Cifar, self).__init__(
            conv1 = L.Convolution2D(3, 32, 3, pad=1),
            conv2 = L.Convolution2D(32, 32, 3, pad=1),
            l1 = L.Linear(None, 512),
            l2 = L.Linear(None, 5),
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
    def __init__(self, n=500, eta=0.001, eps=1e-4, weight=1e-4):
        self.n = n
        self.eta = eta
        self.eps = eps
        self.weight = weight
    
    @property
    def lr(self):
        lr = self.eta / np.power(1. + self.t, 0.55)
        if lr > self.eps:
        	return lr
        else:
        	return self.eps
    
    def update_one_cpu(self, param, state):
        g = param.grad
        param.data -= 0.5 * self.lr * (self.n * g + self.weight * param.data) + np.random.normal(0, self.lr, g.shape)

model = O.Classifier(Cifar())
optimizer = SGLD()
optimizer.setup(model)

#updater = training.StandardUpdater(train_iter, optimizer)
#trainer = training.Trainer(updater, (10, 'epoch'), 'HEL')

#trainer.extend(extensions.Evaluator(test_iter, model))
#trainer.extend(extensions.LogReport())
#trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
#trainer.extend(extensions.ProgressBar())
#trainer.extend(O.Validation(train_entr_iter, model))
#trainer.extend(O.BysAccuracy(test_target))

#trainer.run()

p = np.load('HEL/bayesian.npy').astype(np.float32)
p /= p.sum(axis=1, keepdims=True)
y = p.argmax(axis=1)
p[p < 1e-30] = 1e-30
entropy = -np.sum(p * np.log(p), axis=1)
np.savetxt('SGLDdata.csv', np.vstack([entropy, y, train_target]).T, delimiter=',')

index = np.argsort(entropy)
#S.save_npz('learned_model', model)
#S.save_npz('learned_opt', optimizer)

for i in xrange(1, 5):
    train = chainer.datasets.TupleDataset(train_data[index[-1:-5000*i-1:-1]], train_target[index[-1:-5000*i-1:-1]])
    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, len(test_data), repeat=False, shuffle=False)

    S.load_npz('learned_model', model)
    S.load_npz('learned_opt', optimizer)

    updater = training.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (20, 'epoch'), 'HEL')

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport(log_name='hel_{}'.format(i)))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    #trainer.extend(O.Validation(test_iter, model))
    #trainer.extend(O.BysAccuracy(test_target))

    trainer.run()

train = chainer.datasets.TupleDataset(train_data, train_target)
train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, len(test_data), repeat=False, shuffle=False)

S.load_npz('learned_model', model)
S.load_npz('learned_opt', optimizer)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), 'HEL')

trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.LogReport(log_name='no_hel'))
trainer.extend(extensions.PrintReport(
['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())
#trainer.extend(O.Validation(test_iter, model))
#trainer.extend(O.BysAccuracy(test_target))

trainer.run()