import numpy as np
from chainer.datasets import tuple_dataset

if __name__ ==  '__main__':
    r , s = 8., 1.
    
    theta = np.arange(400) / 400. * np.pi
    right = np.vstack((np.sin(theta) - 0.25, np.cos(theta) + 0.5)) * r
    right = right.T.astype(np.float32)
    image = np.vstack((right, -right)) + np.random.randn(800, 2) * s
    np.save('train.npy', image)
    
    theta = np.arange(200) / 200. * np.pi
    right = np.vstack((np.sin(theta) - 0.25, np.cos(theta) + 0.5)) * r
    right = right.T.astype(np.float32)
    image = np.vstack((right, -right)) + np.random.randn(400, 2) * s
    np.save('test.npy', image)