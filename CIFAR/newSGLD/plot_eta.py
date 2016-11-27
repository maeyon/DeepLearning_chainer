import json
import numpy as np
import matplotlib.pyplot as plt

for i in [0.6, 0.7, 0.8, 0.9]:
    f = open("log_{}".format(i), "r")
    data = json.load(f)
    f.close()
    
    acc = []
    
    for j in xrange(len(data)):
        acc.append(data[j]["validation/main/accuracy"])
    
    epoch = np.arange(len(data)) + 1
    acc = np.array(acc)
    
    plt.plot(epoch, acc, label="eta = {}".format(i))
    plt.legend(loc="upper left")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracy with good eta")
    plt.xlim(0, len(data) + 1)
    plt.ylim(0, 1)

plt.show()