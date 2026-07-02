import numpy as np

def retirement_risk(age, k = 2.5):
    return np.exp((age - 35) / k)

def evolve_func(rating, age):
    z = -0.005 * (age - 27) ** 3
    # rating += np.random.normal(z, 0.1 * (rating + 0.001))
    rating += np.random.normal(z, 0.5)
    return rating

def reward_func(position, k = 0.3):
    return np.exp(-k * (position- 1))