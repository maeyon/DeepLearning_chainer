import numpy as np
from chainer.datasets import tuple_dataset

def get_semiarch(r=8., s=1.):
    matrix = np.random.rand(2, 500).astype(np.float32) * 2. - 1.
    
    theta = np.arange(2000) / 2000. * np.pi
    right = np.vstack((np.sin(theta) - 0.25, np.cos(theta) + 0.5)) * r
    right = right.T.astype(np.float32)
    image = np.vstack((right, -right)) + np.random.randn(4000, 2) * s
    np.save('train.npy', image)
    data = image.dot(matrix).astype(np.float32)
    label = np.hstack((np.zeros(2000), np.ones(2000))).astype(np.int32)
    train = tuple_dataset.TupleDataset(data, label)
    
    theta = np.arange(500) / 500. * np.pi
    right = np.vstack((np.sin(theta) - 0.25, np.cos(theta) + 0.5)) * r
    right = right.T.astype(np.float32)
    image = np.vstack((right, -right)) + np.random.randn(1000, 2) * s
    np.save('test.npy', image)
    data = image.dot(matrix).astype(np.float32)
    label = np.hstack((np.zeros(500), np.ones(500))).astype(np.int32)
    test = tuple_dataset.TupleDataset(data, label)
    
    return train, test