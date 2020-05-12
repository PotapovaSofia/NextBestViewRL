import math
from IPython.display import clear_output
import matplotlib.pyplot as plt

import numpy as np


def build_epsilon_func(epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=10000):
    return lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * \
        math.exp(-1. * frame_idx / epsilon_decay)


def plot(save_path, frame_idx, rewards, novps, losses):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('number of vp: %s' % np.mean(novps[-10:]))
    plt.plot(novps)
    plt.subplot(133)
    plt.title('loss')
    if len(losses) > 20:
        plt.plot(losses[20:])
    else:
        plt.plot(losses)
    plt.savefig(save_path)
