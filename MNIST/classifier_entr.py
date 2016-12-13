import numpy as np
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter

class ClassifierEntr(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(ClassifierEntr, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.entr = None

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
        print 'y =', self.y[0]
        print 't =', t[0]
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss