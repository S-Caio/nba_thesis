"""
Everything about a player's life outside a specific contract: aging each
season, retiring, entering as a rookie, and the post-retirement rookie
draft that assigns new entrants to teams in draft order.

NOTE: `evolve_func` and `retirement_risk` aren't redefined here -- they're
your existing rating-curve / retirement-odds functions. Import them from
wherever you already keep them (e.g. `from .player_curves import
evolve_func, retirement_risk`).
"""
import numpy as np
from scipy import stats

from .constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY
from .state import LeagueState
from .rosters import get_team_counts

# Plug in your real implementations here:
from .player_curves import evolve_func, retirement_risk


def player_update(state: LeagueState) -> None:
    """Age every player by one season and evolve their rating accordingly.

    IMPORTANT: `.copy()` is required here. `state.players[:, AGE]` is a
    view, not a copy -- writing the incremented age back into the array
    before evolve_func runs would make every player evolve using their
    *post-increment* age instead of the age they were that season.
    """
    ratings = state.players[:, RATING].copy()
    age = state.players[:, AGE].copy()

    state.players[:, AGE] = age + 1
    state.players[:, RATING] = [evolve_func(r, a) for r, a in zip(ratings, age)]
    state.players[:, RATING] = np.clip(state.players[:, RATING], 0.1, np.inf)


def new_entrants(n_new_players: int, age_mean: float = 21, age_std: float = 1,
                  rating_shape: float = 0.8) -> np.ndarray:
    ratings = stats.lognorm.rvs(loc=0, s=rating_shape, size=n_new_players)
    teams = np.zeros(n_new_players)
    ages = np.clip(np.round(np.random.normal(age_mean, age_std, size=n_new_players)), 19, 40)
    contract_lens = np.zeros(n_new_players)
    salaries = np.zeros(n_new_players)

    entrants = np.vstack([ratings, teams, ages, contract_lens, salaries]).T
    return entrants[np.argsort(-entrants[:, RATING])]


def pick_retirees(state: LeagueState, n_to_retire: int = 60) -> np.ndarray:
    scores = retirement_risk(state.players[:, AGE])
    probs = scores / np.sum(scores)
    return np.random.choice(len(state.players), size=n_to_retire, p=probs, replace=False)


def churn_player_pool(state: LeagueState, n_to_retire: int = 60) -> np.ndarray:
    """
    Retire a weighted-random batch of players (older players more likely)
    and replace them in-place with freshly generated rookies.

    Returns the array indices of the newly inserted rookies -- the same
    indices the retirees occupied, since insertion overwrites those rows
    directly. A separate step (assign_rookie_draft) decides which teams
    those rookies go to; this function only concerns itself with pool
    turnover, not team assignment.

    Note: there's no meaningful "just delete the retirees" step separate
    from insertion -- retirees' data isn't preserved anywhere, so removal
    and replacement happen as one vectorized write.
    """
    retire_idx = pick_retirees(state, n_to_retire=n_to_retire)
    entrants = new_entrants(n_new_players=n_to_retire)
    state.players[retire_idx] = entrants
    return retire_idx


def assign_rookie_draft(state: LeagueState, config: LeagueConfig,
                         full_draft_order: list[str], rookie_idx: np.ndarray) -> None:
    """
    Assign freshly entered rookies (at `rookie_idx` in state.players) to
    teams by draft order, across two rounds. Teams with a full roster or
    insufficient cap space forfeit that pick -- the rookie stays on the
    open market instead of being forced on.

    Deliberately takes `rookie_idx` as an argument rather than computing
    it itself: this makes the draft mechanics (the part with the cap/roster
    guard logic) testable with a fixed, known set of rookie slots, with no
    dependency on retirement randomness.
    """
    # Sort rookie indices based on their rating in the player matrix (highest to lowest)
    rookie_ratings = state.players[rookie_idx, RATING]
    sorted_rookie_idx = rookie_idx[np.argsort(-rookie_ratings)]

    n_rookies = len(sorted_rookie_idx)
    team_counts = get_team_counts(state, config)

    pick_num = 0
    for _round_num in range(2):
        for agent in full_draft_order:
            if pick_num >= n_rookies:
                break

            is_roster_full = team_counts[agent] >= config.players_per_team
            is_cap_broken = state.team_salaries[agent] + config.rookie_salary > config.salary_cap

            if is_roster_full or is_cap_broken:
                pick_num += 1
                continue

            player_idx = sorted_rookie_idx[pick_num]
            agent_numeric_idx = int(agent.split("_")[1])
            state.players[player_idx, TEAM] = agent_numeric_idx + 1
            state.players[player_idx, CONTRACT_LEN] = config.rookie_contract_len
            state.players[player_idx, SALARY] = config.rookie_salary

            team_counts[agent] += 1
            state.team_salaries[agent] += config.rookie_salary

            pick_num += 1


def run_rookie_draft(state: LeagueState, config: LeagueConfig, full_draft_order: list[str],
                      n_to_retire: int = 60) -> None:
    """Thin orchestrator, kept for backward compatibility with env.py's
    existing call site -- same signature and behavior as before."""
    rookie_idx = churn_player_pool(state, n_to_retire=n_to_retire)
    assign_rookie_draft(state, config, full_draft_order, rookie_idx)
