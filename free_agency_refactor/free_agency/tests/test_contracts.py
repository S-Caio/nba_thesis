"""
Tests for contracts.py. These build a small, fully-controlled LeagueState
by hand rather than going through the env, so each guard clause can be
tested in isolation with no randomness involved.
"""
import numpy as np
import pytest

from free_agency.constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY
from free_agency.state import LeagueState
from free_agency.contracts import handle_signing, contract_update


def make_state(n_players=4, players_per_team=2, agents=("team_0", "team_1")):
    """Minimal deterministic league: 4 players, ratings 4,3,2,1, all free agents."""
    players = np.array([
        [4.0, 0, 25, 0, 0],
        [3.0, 0, 26, 0, 0],
        [2.0, 0, 27, 0, 0],
        [1.0, 0, 28, 0, 0],
    ])[:n_players]
    teams = {a: np.zeros(players_per_team) for a in agents}
    team_salaries = {a: 0.0 for a in agents}
    team_win_pct = {a: 0.5 for a in agents}
    team_has_history = {a: 0.0 for a in agents}
    return LeagueState(players, teams, team_salaries, team_win_pct, team_has_history)


def make_config():
    return LeagueConfig(salary_cap=20.0,
                         salary_ranges=np.array([0, 5, 10, 15]),
                         contract_lengths=np.array([1, 2, 3]))


# --- handle_signing ------------------------------------------------------

def test_successful_signing_updates_player_team_and_roster():
    state = make_state()
    config = make_config()

    # player_id=0, salary_idx=1 (5), contract_len_idx=0 (1 year)
    handle_signing(state, config, "team_0", (0, 1, 0))

    assert state.players[0, TEAM] == 1  # team_0 -> numeric idx 0 -> marker 1
    assert state.players[0, SALARY] == 5
    assert state.players[0, CONTRACT_LEN] == 1
    assert state.team_salaries["team_0"] == 5
    assert state.teams["team_0"][0] == 4.0  # rating landed in first empty slot


def test_signing_already_owned_player_is_a_no_op():
    state = make_state()
    config = make_config()
    state.players[0, TEAM] = 2  # already on team_1

    handle_signing(state, config, "team_0", (0, 1, 0))

    assert state.team_salaries["team_0"] == 0
    assert state.teams["team_0"][0] == 0.0


def test_signing_that_would_break_cap_is_a_no_op():
    state = make_state()
    config = make_config()
    state.team_salaries["team_0"] = 18.0  # cap is 20

    handle_signing(state, config, "team_0", (0, 1, 0))  # offers 5 -> would total 23

    assert state.players[0, TEAM] == 0  # still a free agent
    assert state.team_salaries["team_0"] == 18.0  # unchanged


def test_signing_onto_full_roster_is_a_no_op():
    state = make_state(players_per_team=2)
    config = make_config()
    state.teams["team_0"] = np.array([9.0, 9.0])  # both slots occupied

    handle_signing(state, config, "team_0", (0, 1, 0))

    assert state.players[0, TEAM] == 0
    assert state.team_salaries["team_0"] == 0


def test_signing_fills_first_empty_slot_not_first_slot():
    state = make_state(players_per_team=2)
    config = make_config()
    state.teams["team_0"] = np.array([9.0, 0.0])  # slot 0 taken, slot 1 free

    handle_signing(state, config, "team_0", (0, 1, 0))

    assert state.teams["team_0"][0] == 9.0  # untouched
    assert state.teams["team_0"][1] == 4.0  # new signing landed here


# --- contract_update -------------------------------------------------------

def test_contract_update_decrements_active_contracts():
    state = make_state()
    state.players[0, TEAM] = 1
    state.players[0, CONTRACT_LEN] = 3

    contract_update(state)

    assert state.players[0, CONTRACT_LEN] == 2
    assert state.players[0, TEAM] == 1  # still signed


def test_contract_update_releases_expired_players():
    state = make_state()
    state.players[0, TEAM] = 1
    state.players[0, CONTRACT_LEN] = 1  # will hit 0 this call
    state.players[0, SALARY] = 10

    contract_update(state)

    assert state.players[0, CONTRACT_LEN] == 0
    assert state.players[0, TEAM] == 0  # released to free agency
    assert state.players[0, SALARY] == 0  # salary cleared


def test_contract_update_ignores_players_never_signed():
    state = make_state()  # all free agents, CONTRACT_LEN == 0, TEAM == 0

    contract_update(state)

    assert np.all(state.players[:, CONTRACT_LEN] == 0)
    assert np.all(state.players[:, TEAM] == 0)
