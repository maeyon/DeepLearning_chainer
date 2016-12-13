import json
import numpy as np
import matplotlib.pyplot as plt

f = open("log_5", "r")
data = json.load(f)
f.close()

epoch = np.arange(len(data)) + 1
tra_loss = []
val_loss = []
tra_acc = []
val_acc = []

for i in xrange(len(data)):
    tra_loss.append(data[i]["main/loss"])
    val_loss.append(data[i]["validation/main/loss"])
    tra_acc.append(data[i]["main/accuracy"])
    val_acc.append(data[i]["validation/main/accuracy"])

tra_loss = np.array(tra_loss)
val_loss = np.array(val_loss)
tra_acc = np.array(tra_acc)
val_acc = np.array(val_acc)

plt.subplot(1, 2, 1)
plt.plot(epoch, tra_loss, label="train")
plt.plot(epoch, val_loss, label="validation")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss function")
plt.xlim(0, len(data) + 1)
plt.ylim(0, )

plt.subplot(1, 2, 2)
plt.plot(epoch, tra_acc, label="train")
plt.plot(epoch, val_acc, label="validation")
plt.legend(loc="lower right")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("accuracy rate")
plt.xlim(0, len(data) + 1)
plt.ylim(0, 1)

plt.tight_layout()

plt.show()