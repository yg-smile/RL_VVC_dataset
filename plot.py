import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import numpy as np

rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 12})
rc('text', usetex=True)


def smooth(y, box_pts):
    # smooth curves by moving average
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def plot_res(envs, algos, metric, smoothing,
             ylabel, xlabel):

    fig, axes = plt.subplots(nrows=1, ncols=len(envs), figsize=(4.*len(envs), 4.))

    for ax, env in enumerate(envs):
        res_all = []
        for a in algos:
            with open('./res/data/{}_{}.pkl'.format(env, a), 'rb') as f:
                res_all.append(pickle.load(f))

        v_all = []
        for res in res_all:
            if res[metric]:
                v_all.append(np.array(res[metric]))
            else:
                v_all.append([])

        for i, v_each in enumerate(v_all):
            if isinstance(v_each, np.ndarray):
                axes[ax].plot(smooth(np.percentile(v_each, q=50, axis=0), smoothing), label=algos[i])
                axes[ax].fill_between(x=np.arange(v_each.shape[1] - smoothing + 1),
                                      y1=smooth(np.percentile(v_each, q=10, axis=0), smoothing),
                                      y2=smooth(np.percentile(v_each, q=90, axis=0), smoothing), alpha=0.4)
        axes[ax].grid(True, alpha=0.1)
        if env == '8500':
            axes[ax].title.set_text('{} node'.format(env))
        else:
            axes[ax].title.set_text('{} bus'.format(env))
        if ax == 0:
            axes[ax].set_ylabel(ylabel)
        if ax == 1:
            axes[ax].set_xlabel(xlabel)

    plt.legend()
    plt.savefig('./res/figs/bus{}_{}.pdf'.format(envs, metric), bbox_inches='tight')
    plt.show()
