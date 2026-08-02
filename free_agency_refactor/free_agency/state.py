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
    players: np.ndarray
    teams: dict[str, np.ndarray]
    team_salaries: dict[str, float]
    team_win_pct: dict[str, float]
    team_has_history: dict[str, float]
    team_win_pct_history: dict[str, np.ndarray]   # NEW: fixed shape (K,) per agent
    team_rating_history: dict[str, np.ndarray]    # NEW: fixed shape (K,) per agent

    offer_player: np.ndarray
    offer_salary_idx: np.ndarray
    offer_length_idx: np.ndarray


def new_league_state(config: LeagueConfig, possible_agents: list[str]) -> LeagueState:
    players = _generate_players(config.n_players)
    teams = {agent: np.zeros(config.players_per_team) for agent in possible_agents}
    team_salaries = {agent: 0.0 for agent in possible_agents}
    team_win_pct = {agent: 0.5 for agent in possible_agents}
    team_has_history = {agent: 0.0 for agent in possible_agents}
    team_win_pct_history = {agent: np.zeros(config.history_window, dtype=np.float32) for agent in possible_agents}
    team_rating_history = {agent: np.zeros(config.history_window, dtype=np.float32) for agent in possible_agents}

    offer_player = -np.ones(config.n_teams, dtype=np.int64)
    offer_salary_idx = -np.ones(config.n_teams, dtype=np.int64)
    offer_length_idx = -np.ones(config.n_teams, dtype=np.int64)
    return LeagueState(
        players, teams, team_salaries, team_win_pct, team_has_history,
        team_win_pct_history, team_rating_history,
        offer_player, offer_salary_idx, offer_length_idx
    )


def record_season_history(league: LeagueState, agent: str, win_pct: float, team_rating: float) -> None:
    """Shift the fixed-size window left and drop the new value in the last slot.
    Called once per agent per season — not per step — so np.roll's allocation
    cost here is irrelevant."""
    league.team_win_pct_history[agent] = np.roll(league.team_win_pct_history[agent], -1)
    league.team_win_pct_history[agent][-1] = win_pct

    league.team_rating_history[agent] = np.roll(league.team_rating_history[agent], -1)
    league.team_rating_history[agent][-1] = team_rating

    league.team_has_history[agent] = 1.0

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

def agent_to_team_id(agent: str) -> int:
    """The value stored in players[:, TEAM] for this agent's roster.
    1-indexed; 0 (FREE_AGENT_MARKER) means unsigned."""
    return int(agent.split("_")[1]) + 1