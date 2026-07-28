"""
LeagueState: the single object that carries everything the env would
otherwise scatter across self.players / self.teams / self.team_salaries /
self.team_win_pct / self.team_has_history.

Bundling these together makes the data flow explicit: any function that
needs to read or mutate league data takes a LeagueState, full stop. No more
guessing which of five parallel self.* dicts a given method touches.
"""
from dataclasses import dataclass
import numpy as np
from scipy import stats

from .constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY, OFFERS


@dataclass
class LeagueState:
    players: np.ndarray                    # (n_players, 5)
    teams: dict[str, np.ndarray]            # agent -> (players_per_team,) ratings
    team_salaries: dict[str, float]
    team_win_pct: dict[str, float]
    team_has_history: dict[str, float]

    offer_player: np.ndarray       # shape (n_teams,), dtype=int64
    offer_salary_idx: np.ndarray   # shape (n_teams,), dtype=int64
    offer_length_idx: np.ndarray   # shape (n_teams,), dtype=int64


def _generate_players(n_players: int, age_mean: float = 27, age_std: float = 4,
                       rating_shape: float = 1.0) -> np.ndarray:
    ratings = sorted(stats.lognorm.rvs(loc=0, s=rating_shape, size=n_players), reverse=True)
    zero_array = np.zeros(n_players) # Used for teams, contract_lens, salaries, and offers
    # teams = np.zeros(n_players)
    ages = np.clip(np.round(np.random.normal(age_mean, age_std, size=n_players)), 19, 40)
    # contract_lens = np.zeros(n_players)
    # salaries = np.zeros(n_players)

    return np.vstack([
        ratings, 
        zero_array, # Teams 
        ages, 
        zero_array, # Contract lens
        zero_array, # Salaries
        zero_array, # Offers
        ]).T


def new_league_state(config: LeagueConfig, possible_agents: list[str]) -> LeagueState:
    """Fresh league: new player pool, empty rosters, neutral win pct/history."""
    players = _generate_players(config.n_players)
    teams = {agent: np.zeros(config.players_per_team) for agent in possible_agents}
    team_salaries = {agent: 0.0 for agent in possible_agents}
    team_win_pct = {agent: 0.5 for agent in possible_agents}
    team_has_history = {agent: 0.0 for agent in possible_agents}

    offer_player=-np.ones(config.n_teams, dtype=np.int64)
    offer_salary_idx=-np.ones(config.n_teams, dtype=np.int64)
    offer_length_idx=-np.ones(config.n_teams, dtype=np.int64)
    return LeagueState(
        players, 
        teams, 
        team_salaries,
        team_win_pct,
        team_has_history, 
        offer_player, 
        offer_salary_idx, 
        offer_length_idx
        )
