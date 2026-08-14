import numpy as np

def retirement_risk(age, k = 2.5):
    return np.exp((age - 35) / k)

def evolve_func(rating, age):
    z = -0.005 * (age - 27) ** 3
    # rating += np.random.normal(z, 0.1 * (rating + 0.001))
    rating += np.random.normal(z, 0.5)
    return rating

def evolve_v2(rating, age, potential, noise_scale=0.14):
    if age < 27:
        drift = 0.015 * (27 - age) + 0.03 * (potential - 1.0)      # up to +0.12 log-units at 19, tapering to 0 at 27
    else:
        drift = -0.02 * (age - 27)      # increasingly negative post-peak
    noise = np.random.standard_normal() * noise_scale
    return rating * np.exp(drift + noise)

def evolve_v4(rating, age, potential, kappa=0.35, sigma=0.14, df=4):
    x = np.log(rating)
    target = np.log(age_curve(age)) + np.log(potential)   # this player's own expected level
    x_new = x + kappa * (target - x) + np.random.standard_normal() * sigma
    return np.exp(x_new)

def age_curve(age):
    if age < 27:
        return 2.0 * np.exp(-0.03 * (27 - age))
    return 2.0 * np.exp(-0.025 * (age - 27))


def reward_func(position, k = 0.15):
    return np.exp(-k * (position- 1))