import numpy as np
from chainer.datasets import tuple_dataset

if __name__ == '__main__':
    s = 0.5
    
    theta = (np.arange(200) / 200. - 0.5) * 2.
    right = np.vstack((1. / np.cos(theta) + 1., np.tan(theta))).T.astype(np.float32)
    up = np.vstack((np.tan(theta), 1. / np.cos(theta) + 1.)).T.astype(np.float32)
    image = np.vstack((right, up, -right, -up)) + np.random.randn(800, 2) * s
    np.save('train.npy', image)
    
    theta = (np.arange(100) / 100. - 0.5) * 2.
    right = np.vstack((1. / np.cos(theta) + 1., np.tan(theta))).T.astype(np.float32)
    up = np.vstack((np.tan(theta), 1. / np.cos(theta) + 1.)).T.astype(np.float32)
    image = np.vstack((right, up, -right, -up)) + np.random.randn(400, 2) * s
    np.save('test.npy', image)