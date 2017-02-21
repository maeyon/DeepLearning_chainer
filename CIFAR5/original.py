import numpy as np
from scipy.sparse import coo_matrix
import chainer.functions as F
import chainer.optimizer
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer.variable import Variable
from chainer import reporter
import copy

class Classifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.entr = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
    
    def forward(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        y = self.predictor(*x).data
        I = np.arange(len(t))
        J = y.argmax(axis=1)
        V = np.ones(len(t))
        return coo_matrix((V, (I, J)), shape=y.shape)
        
class Validation(chainer.training.extensions.Evaluator):
    trigger = 100, 'iteration'
    default_name = 'validation'
    
    def __init__(self, iterator, target):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target
        
    def __call__(self, trainer=None):
        iterator = self._iterators['main']
        target = self._targets['main']
        
        it = copy.copy(iterator)

        for batch in it:
            in_arrays = chainer.dataset.convert.concat_examples(batch, None)
            if isinstance(in_arrays, tuple):
                in_vars = tuple(Variable(x, volatile='on') for x in in_arrays)
                np.save('bayesian.npy', np.load('bayesian.npy') + target.forward(*in_vars))
            elif isinstance(in_arrays, dict):
                in_vars = {key: Variable(x, volatile='on') for key, x in iteritems(in_arrays)}
                np.save('bayesian.npy', np.load('bayesian.npy') + target.forward(**in_vars))
            else:
                in_vars = Variable(in_arrays, volatile='on')
                np.save('bayesian.npy', np.load('bayesian.npy') + target.forward(in_vars))
                
class BysAccuracy(chainer.training.extensions.Evaluator):
    trigger = 1, 'epoch'
    default_name = 'validation'
    
    def __init__(self, label):
        self.t = label
        
    def __call__(self, trainer=None):
        p = np.load('bayesian.npy').astype(np.float32)
        ac = (p.argmax(axis=1) == self.t).mean()
        with open('accuracy.csv', 'a') as f:
            f.write('{}\n'.format(ac))