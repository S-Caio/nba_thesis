"""
Everything about a player's contract lifecycle: signing a free agent,
and the annual decay/expiration pass.

These are pure-ish functions (they mutate the LeagueState passed in, but
take no hidden dependencies) which makes them trivial to unit test:
build a small fake LeagueState, call the function, assert on the result.
No PettingZoo env required.
"""
import numpy as np

from .constants import LeagueConfig, RATING, TEAM, CONTRACT_LEN, SALARY
from .state import LeagueState


# Constants for action decoding
ACTION_PLAYER_ID = 0
ACTION_SALARY = 1
ACTION_CONTRACT_LENGTH = 2

FREE_AGENT_MARKER = 0


def decode_flat_action(action_idx: int, config: LeagueConfig) -> tuple[int, int, int]:
    """
    Decodes a single flat scalar action back into discrete choices
    without needing to look up a giant array in memory.
    """
    n_salaries = len(config.salary_ranges)
    n_lengths = len(config.contract_lengths)
    
    # Standard multi-dimensional index unraveling
    # Block size for each player combination
    player_stride = n_salaries * n_lengths
    
    player_idx = action_idx // player_stride
    remainder = action_idx % player_stride
    
    salary_idx = remainder // n_lengths
    length_idx = remainder % n_lengths
    
    return int(player_idx), int(salary_idx), int(length_idx)

def make_action_mask(state, config, team):
    team_vec = state.teams[team]
    salary = state.team_salaries[team]

    action_map = config.action_map
    
    # Check to see if signing doesn't exceed salary cap
    salary_values = config.salary_ranges[action_map[:, ACTION_SALARY]]
    salary_mask = (
        salary + salary_values <= config.salary_cap
    )
    
    # Check to see if signing the player doesn't take the team above the maximum number of players
    num_players = len(team_vec[team_vec > 0.0])
    if num_players + 1 <= config.players_per_team:
        num_players_mask = np.ones_like(salary_mask)
    else:
        num_players_mask = np.zeros_like(salary_mask)

    # Check to see if player is in the market
    player_id_actions = action_map[:, ACTION_PLAYER_ID].astype(int)
    available_players_mask = np.where(state.players[player_id_actions, TEAM] == FREE_AGENT_MARKER, 1, 0)

    mask_all = num_players_mask & salary_mask & available_players_mask
    if num_players >= (config.players_per_team - 1):
        mask_all = np.append(mask_all, 1)
    else:
        mask_all = np.append(mask_all, 0)

    return mask_all


def handle_signing(state: LeagueState, config: LeagueConfig, agent: str, action) -> None:
    """
    Attempt to sign a free agent for `agent`. `action` is (player_id,
    salary_idx, contract_len_idx) as decoded from the MultiDiscrete action.
    No-ops (silently) if any guard clause fails -- mirrors the original
    "wasted turn" semantics.
    """

    # Skip if action is the "NULL" action
    if action == config.n_proper_actions:
        return

    player_id, salary_idx, contract_len_idx = decode_flat_action(action, config)
    offered_salary = config.salary_ranges[salary_idx]
    chosen_length = config.contract_lengths[contract_len_idx]

    # --- GUARD CLAUSES ---
    if state.players[player_id, TEAM] != FREE_AGENT_MARKER:
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
