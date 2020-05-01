import matplotlib.pyplot as plt
import numpy as np

def PlotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    run_avg = np.empty(N)
    for t in range(N):
        run_avg[t] = np.mean(scores[max(0, t-window):(t+1)])

    if x is None:
        x = [i for i in range(N)]
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(x, run_avg)
    plt.savefig(filename)
