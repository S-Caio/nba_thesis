"""
Reward function: exponentially decreasing
"""
import numpy as np

def reward_func(position, k = 0.3):
    return np.exp(-k * (position- 1))
