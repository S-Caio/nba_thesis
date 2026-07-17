"""
Shared constants and configuration for the free agency environment.

Centralizing these two things — column layout and tunable parameters —
means changing "how big is the salary cap" or "which column is age"
happens in exactly one place instead of being hunted down across methods.
"""
from dataclasses import dataclass, field
import numpy as np


# --- Player matrix column layout ---------------------------------------
# self.players is an (n_players, 5) array. These names replace magic
# indices like `players[:, 1]` with `players[:, TEAM]`.
RATING = 0
TEAM = 1
AGE = 2
CONTRACT_LEN = 3
SALARY = 4

N_PLAYER_COLS = 5


@dataclass
class LeagueConfig:
    n_teams: int = 30
    players_per_team: int = 10
    salary_cap: float = 100.0
    n_seasons: int = 10

    # Signing action space
    salary_ranges: np.ndarray = field(default_factory=lambda: np.arange(0, 35, 5))
    contract_lengths: np.ndarray = field(default_factory=lambda: np.arange(1, 6))

    # Rookie draft
    rookie_contract_len: int = 4
    rookie_salary: int = 5
    n_new_entrants_per_season: int = 60

    # Season simulation
    season_noise_scale: float = 10.0

    action_map: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        grid = np.mgrid[
            0:self.n_players,
            0:len(self.salary_ranges),
            0:len(self.contract_lengths)
        ]

        self.action_map = grid.reshape(3, -1).T

    @property
    def n_players(self) -> int:
        return self.n_teams * self.players_per_team

    @property
    def n_proper_actions(self) -> int:
        return self.n_players * len(self.salary_ranges) * len(self.contract_lengths)
