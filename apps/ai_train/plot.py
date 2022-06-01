import numpy as np
import matplotlib.pyplot as plt
import os, glob

def saveFig(test, pred, label, prefix="", fig_count=10):
    """
    Save plot graph
    """
    if not os.path.exists(F"./ai_models/fig/{prefix}"):
        os.makedirs(F"./ai_models/fig/{prefix}")

    for f in glob.glob(F"./ai_models/fig/{prefix}/{prefix}-*.jpg"):
        os.remove(f)

    total = len(test)
    step = total // fig_count
    for i in range(0, total, step):
        plt.clf()
        plt.plot(np.arange(len(test[i])), test[i], label="input")
        plt.plot(np.arange(len(test[i]), len(test[i]) + len(pred[i])), label[i], label="label")
        plt.plot(np.arange(len(test[i]), len(test[i]) + len(pred[i])), pred[i], label="predit")
        plt.legend()
        plt.savefig(F"./ai_models/fig/{prefix}/{prefix}-{i}.jpg")
