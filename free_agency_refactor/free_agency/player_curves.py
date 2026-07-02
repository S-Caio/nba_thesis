"""
Placeholder rating-curve functions. These weren't in the code you shared,
so these are stand-ins purely so the package can run end-to-end and be
tested. Swap for your real implementations.
"""
import numpy as np


def evolve_func(rating: float, age: float) -> float:
    """Toy aging curve: peak around 27, gentle decline either side."""
    age_factor = 1.0 - 0.01 * abs(age - 27)
    noise = np.random.normal(0, 0.03)
    return rating * age_factor + noise


def retirement_risk(ages: np.ndarray) -> np.ndarray:
    """Toy retirement odds: risk increases sharply past 33."""
    return np.clip((ages - 25) ** 2, 1, None)
