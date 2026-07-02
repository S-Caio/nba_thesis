"""
Placeholder reward shaping. Swap for your real reward_func.
"""


def reward_func(position: int, k: float = 1.0) -> float:
    """Toy reward: higher for better draft/standing position, scaled by k."""
    return k * (31 - position) / 30
