"""
Everything about a player's contract lifecycle: signing a free agent,
and the annual decay/expiration pass.

These are pure-ish functions (they mutate the LeagueState passed in, but
take no hidden dependencies) which makes them trivial to unit test:
build a small fake LeagueState, call the function, assert on the result.
No PettingZoo env required.
"""
import numpy as np

from .constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY
from .state import LeagueState


def handle_signing(state: LeagueState, config: LeagueConfig, agent: str, action) -> None:
    """
    Attempt to sign a free agent for `agent`. `action` is (player_id,
    salary_idx, contract_len_idx) as decoded from the MultiDiscrete action.
    No-ops (silently) if any guard clause fails -- mirrors the original
    "wasted turn" semantics.
    """
    player_id, salary_idx, contract_len_idx = action
    offered_salary = config.salary_ranges[salary_idx]
    chosen_length = config.contract_lengths[contract_len_idx]

    # --- GUARD CLAUSES ---
    if state.players[player_id, TEAM] != 0:
        return  # already signed elsewhere

    if state.team_salaries[agent] + offered_salary > config.salary_cap:
        return  # would break the cap

    team_vector = state.teams[agent]
    empty_slots = np.where(team_vector == 0.0)[0]
    if len(empty_slots) == 0:
        return  # roster full

    # --- EXECUTE SIGNING ---
    first_empty_slot = empty_slots[0]
    state.teams[agent][first_empty_slot] = state.players[player_id, RATING]
    state.team_salaries[agent] += offered_salary

    agent_numeric_idx = int(agent.split("_")[1])
    state.players[player_id, TEAM] = agent_numeric_idx + 1
    state.players[player_id, CONTRACT_LEN] = chosen_length
    state.players[player_id, SALARY] = offered_salary


def contract_update(state: LeagueState) -> None:
    """Dwindle every signed player's remaining contract length by one season;
    release anyone whose contract just hit zero back to the open market."""
    signed_mask = state.players[:, CONTRACT_LEN] > 0
    state.players[signed_mask, CONTRACT_LEN] -= 1

    expired_mask = (state.players[:, CONTRACT_LEN] == 0) & (state.players[:, TEAM] != 0)
    state.players[expired_mask, TEAM] = 0.0
    state.players[expired_mask, SALARY] = 0.0
