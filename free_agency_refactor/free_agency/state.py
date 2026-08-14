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

from .constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY, OFFERS, POTENTIAL


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
                      rating_loc: float = 1.0, rating_shape: float = 0.7, potential_loc = 0.0, potential_shape = 0.6) -> np.ndarray:
    ratings = sorted(stats.lognorm.rvs(loc=rating_loc, s=rating_shape, size=n_players), reverse=True)

    potential = stats.lognorm.rvs(loc = potential_loc, s = potential_shape, size = n_players)
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
        potential,
        ]).T

def agent_to_team_id(agent: str) -> int:
    """The value stored in players[:, TEAM] for this agent's roster.
    1-indexed; 0 (FREE_AGENT_MARKER) means unsigned."""
    return int(agent.split("_")[1]) + 1


def pick_initial_players(config):
    n_per_team = config.players_per_team // 2
    n_to_pick = n_per_team * config.n_teams

    idx = np.arange(1, config.n_players + 1)
    alpha = 0.3
    score = idx ** (-alpha)
    p = score / score.sum()

    players_to_assign = np.random.choice(range(config.n_players), size = n_to_pick, replace = False, p = p)

    return sorted(players_to_assign), n_per_team


def initial_endowment(state, config):
    players_to_assign, n_per_team = pick_initial_players(config)
    picked_players = state.players[players_to_assign]
    n_picked_players = len(picked_players)

    # How many players go to each team
    while True:
        repeated = np.repeat(np.arange(1, config.n_teams + 1), n_per_team)
        counts = np.random.multinomial(n_picked_players, 
                                np.ones(config.n_teams) / config.n_teams)

        # print(counts)
        if np.all(counts <= config.players_per_team): 
            break

    team_ids = np.repeat(np.arange(1, config.n_teams + 1), counts)
    np.random.shuffle(team_ids)

    # Assigning players to teams
    state.players[players_to_assign, TEAM] = team_ids

    p = [0.325, 0.325, 0.3, 0.05, 0.0] if len(config.contract_lengths) == 5 else [0.325, 0.325, 0.3, 0.05]
    # Assigning contract lengths to players (bias towards shorter contracts)
    c_lengths = np.random.choice(config.contract_lengths,
                    size = n_picked_players,
                    replace = True,
                    p = p
    )
    state.players[players_to_assign, CONTRACT_LEN] = c_lengths

    # Assigning salaries to players (proportional to rating)
    rating = picked_players[:, RATING]
    min_rating = np.min(rating)
    max_rating = np.max(rating)
    rating_percentile = (rating - min_rating) / (max_rating - min_rating)
    expected_salary = rating_percentile * (len(config.salary_ranges) - 1)


    salary_idx = np.arange(len(config.salary_ranges))
    sigma = 1
    weights = np.exp(-0.5 * ((salary_idx[None, :] - expected_salary[:, None]) / sigma) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)

    # print("weights")
    # print(weights)

    salaries = np.array([
        np.random.choice(config.salary_ranges, p=w)
        for w in weights
    ])
    state.players[players_to_assign, SALARY] = salaries



