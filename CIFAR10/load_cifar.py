import sys
import cPickle as pickle
import datetime, math, sys, time

import numpy as np

from chainer import cuda

def unpickle(file):
	import cPickle
	fo = open(file, "rb")
	dict = cPickle.load(fo)
	fo.close()
	return dict

class Data:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def get(self, n, balance=True):
        ind = np.random.permutation(self.data.shape[0])
        if not balance:
            return cuda.to_gpu(self.data[ind[:n],:].astype(np.float32)), cuda.to_gpu(self.label[ind[:n]].astype(np.int32))
        else:
            cnt = [0]*10
            m = 0
            ret_data = np.zeros((n, self.data.shape[1])).astype(np.float32)
            ret_label = np.zeros(n).astype(np.int32)
            for i in range(self.data.shape[0]):
                if cnt[self.label[ind[i]]] < n/10:
                    ret_data[m,:] = self.data[ind[i]]
                    ret_label[m] = self.label[ind[i]]

                    cnt[self.label[ind[i]]] += 1
                    m += 1
                    if m==n:
                        break
            return cuda.to_gpu(ret_data), cuda.to_gpu(ret_label)

    def put(self, data, label):
        if self.data is None:
            self.data = cuda.to_cpu(data)
            self.label = cuda.to_cpu(label)
        else:
            self.data = np.vstack([self.data, cuda.to_cpu(data)])
            self.label = np.hstack([self.label, cuda.to_cpu(label)]).reshape((self.data.shape[0]))

def load_cifar(scale, shift, N_train_labeled, N_train_unlabeled, N_test):
    print 'fetch CIFAR dataset'
    cifar_data = []
    cifar_target = []

    for i in xrange(1, 6):
        d = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
        cifar_data.extend(d["data"])
        cifar_target.extend(d["labels"])

    d = unpickle("cifar-10-batches-py/test_batch")
    for j in xrange(len(d["data"])):
        cifar_data.append(d["data"][j])
        cifar_target.append(d["labels"][j])

    cifar_data = np.array(cifar_data).astype(np.float32).reshape((-1, 3, 32, 32))*scale + shift
    cifar_target = np.array(cifar_target).astype(np.int32)
    
    print len(cifar_target), 'data'

    perm = np.random.permutation(50000)

    # equal number of data in each category
    cnt_l = [0] * 10
    cnt_ul = [0] * 10
    cnt_test = [0] * 10
    ind_l = []
    ind_ul = []
    ind_test = range(50000,60000)

    for i in range(50000):
        l = cifar_target[perm[i]]
        if cnt_l[l] < N_train_labeled/10:
            ind_l.append(perm[i])
            ind_ul.append(perm[i])
            cnt_l[l] += 1
            cnt_ul[l] += 1
        else:
            ind_ul.append(perm[i])
            cnt_ul[l] += 1

    #print cnt_l, cnt_ul, cnt_test
    x_train_l = cifar_data[ind_l]
    x_train_ul = cifar_data[ind_ul]
    x_test = cifar_data[ind_test]
    y_train_l = cifar_target[ind_l]
    y_train_ul = cifar_target[ind_ul]
    y_test = cifar_target[ind_test]

    train_l = Data(x_train_l, y_train_l)
    train_ul = Data(x_train_ul, y_train_ul)
    test_set = Data(x_test, y_test)


    print "load cifar done", train_l.data.shape, train_ul.data.shape, test_set.data.shape
    return train_l, train_ul, test_set