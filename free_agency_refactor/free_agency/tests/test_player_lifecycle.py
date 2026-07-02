import numpy as np
import pytest

from free_agency.constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY
from free_agency.state import LeagueState
from free_agency.player_lifecycle import (
    player_update, pick_retirees, run_rookie_draft,
    churn_player_pool, assign_rookie_draft,
)


def make_state():
    players = np.array([
        [4.0, 0, 20, 0, 0],
        [3.0, 0, 30, 0, 0],
    ])
    return LeagueState(players, {}, {}, {}, {})


def test_player_update_uses_pre_increment_age_not_post_increment():
    """
    Regression test for the aliasing bug: evolve_func must see each
    player's age as it was *before* the +1 increment, not after. We use a
    fake evolve_func that just returns the age it was called with, so we
    can assert directly on what age each player evolved with.
    """
    import free_agency.player_lifecycle as pl

    seen_ages = []

    def fake_evolve(rating, age):
        seen_ages.append(age)
        return rating  # rating unchanged, we only care about age tracking

    original = pl.evolve_func
    pl.evolve_func = fake_evolve
    try:
        state = make_state()
        player_update(state)
    finally:
        pl.evolve_func = original

    assert seen_ages == [20, 30]  # pre-increment ages, not [21, 31]
    assert state.players[:, AGE].tolist() == [21, 31]  # ages themselves did increment


def test_player_update_clips_ratings_at_floor():
    import free_agency.player_lifecycle as pl

    original = pl.evolve_func
    pl.evolve_func = lambda rating, age: -5.0  # force a negative rating
    try:
        state = make_state()
        player_update(state)
    finally:
        pl.evolve_func = original

    assert np.all(state.players[:, RATING] >= 0.1)


def test_pick_retirees_returns_requested_count_with_no_duplicates():
    import free_agency.player_lifecycle as pl

    original = pl.retirement_risk
    pl.retirement_risk = lambda ages: np.ones_like(ages)  # uniform risk
    try:
        state = make_state()
        state.players = np.tile(state.players, (10, 1))  # 20 players
        idx = pick_retirees(state, n_to_retire=5)
    finally:
        pl.retirement_risk = original

    assert len(idx) == 5
    assert len(set(idx)) == 5  # no duplicates (replace=False)


def test_run_rookie_draft_skips_teams_over_cap():
    """A team already at the salary cap should forfeit its rookie pick,
    leaving the rookie unassigned on the market."""
    players = np.array([[1.0, 0, 21, 0, 0]] * 4)
    state = LeagueState(
        players=players,
        teams={"team_0": np.zeros(1), "team_1": np.zeros(1)},
        team_salaries={"team_0": 100.0, "team_1": 0.0},  # team_0 maxed out
        team_win_pct={"team_0": 0.5, "team_1": 0.5},
        team_has_history={"team_0": 0.0, "team_1": 0.0},
    )
    config = LeagueConfig(n_teams=2, players_per_team=1, salary_cap=100.0, rookie_salary=5)

    # n_to_retire=2: team_0's forfeited pick still consumes a slot in the
    # retiree batch, so team_1 needs a second slot available to get its turn.
    run_rookie_draft(state, config, full_draft_order=["team_0", "team_1"], n_to_retire=2)

    assert state.players[:, TEAM].tolist().count(1) == 0  # team_0 got nothing
    assert state.players[:, TEAM].tolist().count(2) == 1  # team_1 got the rookie
    assert state.team_salaries["team_1"] == 5.0


def test_run_rookie_draft_second_round_respects_first_round_salary_additions():
    """
    Regression test for the cap-tracking bug: a team that fills its cap
    room with a round-1 rookie must not be eligible for a round-2 rookie
    in the same draft.
    """
    players = np.array([[1.0, 0, 21, 0, 0]] * 4)
    state = LeagueState(
        players=players,
        teams={"team_0": np.zeros(2)},
        team_salaries={"team_0": 90.0},  # room for exactly one more 5-salary rookie
        team_win_pct={"team_0": 0.5},
        team_has_history={"team_0": 0.0},
    )
    config = LeagueConfig(n_teams=1, players_per_team=2, salary_cap=97.0, rookie_salary=5)

    # full_draft_order has one team; the loop visits it once per round (2 rounds)
    run_rookie_draft(state, config, full_draft_order=["team_0"], n_to_retire=2)

    signed_count = state.players[:, TEAM].tolist().count(1)
    # Without the fix, team_salaries never gets updated inside the loop, so
    # round 2's cap check would incorrectly still see 90.0 and sign a second
    # rookie, pushing the team to 100 -- over its 97 cap.
    assert signed_count == 1  # only round 1's rookie should get signed
    assert state.team_salaries["team_0"] == 95.0  # 90 + one rookie salary, not two


# --- churn_player_pool (pool turnover, isolated from draft assignment) ---

def test_churn_player_pool_returns_indices_where_new_entrants_landed():
    import free_agency.player_lifecycle as pl

    original = pl.retirement_risk
    pl.retirement_risk = lambda ages: np.ones_like(ages)  # uniform risk, deterministic-ish
    try:
        state = make_state()
        state.players = np.tile(state.players, (10, 1))  # 20 players
        rookie_idx = churn_player_pool(state, n_to_retire=5)
    finally:
        pl.retirement_risk = original

    assert len(rookie_idx) == 5
    # every returned slot should now hold a fresh, unsigned rookie
    assert np.all(state.players[rookie_idx, TEAM] == 0)
    assert np.all(state.players[rookie_idx, CONTRACT_LEN] == 0)


def test_churn_player_pool_does_not_touch_non_retired_players():
    import free_agency.player_lifecycle as pl

    original = pl.retirement_risk
    pl.retirement_risk = lambda ages: np.ones_like(ages)
    try:
        state = make_state()
        state.players = np.tile(state.players, (10, 1))
        before = state.players.copy()
        rookie_idx = churn_player_pool(state, n_to_retire=5)
    finally:
        pl.retirement_risk = original

    untouched_idx = [i for i in range(len(state.players)) if i not in rookie_idx]
    assert np.array_equal(state.players[untouched_idx], before[untouched_idx])


# --- assign_rookie_draft (draft mechanics, no retirement randomness needed) ---

def make_rookie_state(n_slots=4, players_per_team=2):
    """Players at indices 0..n_slots-1 are all fresh, unsigned rookies --
    exactly the shape assign_rookie_draft expects to receive."""
    players = np.array([[1.0, 0, 21, 0, 0]] * n_slots)
    teams = {"team_0": np.zeros(players_per_team), "team_1": np.zeros(players_per_team)}
    salaries = {"team_0": 0.0, "team_1": 0.0}
    return LeagueState(players, teams, salaries,
                        {"team_0": 0.5, "team_1": 0.5}, {"team_0": 0.0, "team_1": 0.0})


def test_assign_rookie_draft_assigns_in_draft_order():
    state = make_rookie_state(n_slots=2, players_per_team=1)
    config = LeagueConfig(n_teams=2, players_per_team=1, salary_cap=100.0, rookie_salary=5)

    assign_rookie_draft(state, config, full_draft_order=["team_1", "team_0"],
                         rookie_idx=np.array([0, 1]))

    # team_1 picks first (per full_draft_order), so it gets rookie_idx[0]
    assert state.players[0, TEAM] == 2  # team_1
    assert state.players[1, TEAM] == 1  # team_0


def test_assign_rookie_draft_forfeits_full_roster_without_raising():
    # index 0: pre-existing player already owned by team_0 (fills its 1 slot)
    # index 1, 2: fresh rookie slots -- need two, since a forfeited pick
    # still consumes one slot from the batch (same lesson as before: don't
    # let n_rookies be smaller than a full round of the draft order).
    players = np.array([
        [9.0, 1, 28, 2, 10],   # owned by team_0 already
        [1.0, 0, 21, 0, 0],    # fresh rookie
        [1.0, 0, 22, 0, 0],    # fresh rookie
    ])
    teams = {"team_0": np.array([9.0]), "team_1": np.zeros(1)}
    salaries = {"team_0": 10.0, "team_1": 0.0}
    state = LeagueState(players, teams, salaries,
                         {"team_0": 0.5, "team_1": 0.5}, {"team_0": 0.0, "team_1": 0.0})
    config = LeagueConfig(n_teams=2, players_per_team=1, salary_cap=100.0, rookie_salary=5)

    assign_rookie_draft(state, config, full_draft_order=["team_0", "team_1"],
                         rookie_idx=np.array([1, 2]))

    # team_0's pick (rookie_idx[0] = index 1) is forfeited -- still a free agent
    assert state.players[1, TEAM] == 0
    # team_1's pick (rookie_idx[1] = index 2) succeeds
    assert state.players[2, TEAM] == 2


def test_assign_rookie_draft_leaves_rookie_unassigned_if_all_teams_forfeit():
    players = np.array([
        [9.0, 1, 28, 2, 10],   # owned by team_0 already, fills its 1 slot
        [9.0, 2, 28, 2, 10],   # owned by team_1 already, fills its 1 slot
        [1.0, 0, 21, 0, 0],    # fresh rookie, nobody has room
    ])
    teams = {"team_0": np.array([9.0]), "team_1": np.array([9.0])}
    salaries = {"team_0": 10.0, "team_1": 10.0}
    state = LeagueState(players, teams, salaries,
                         {"team_0": 0.5, "team_1": 0.5}, {"team_0": 0.0, "team_1": 0.0})
    config = LeagueConfig(n_teams=2, players_per_team=1, salary_cap=100.0, rookie_salary=5)

    assign_rookie_draft(state, config, full_draft_order=["team_0", "team_1"],
                         rookie_idx=np.array([2]))

    assert state.players[2, TEAM] == 0  # stays a free agent, no error raised
