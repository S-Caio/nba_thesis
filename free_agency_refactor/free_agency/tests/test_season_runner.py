import numpy as np

from free_agency.season_runner import (
    build_standings, apply_standings, update_win_records, PLAYOFF_CUTOFF, StandingEntry
)
from free_agency.state import LeagueState


def test_build_standings_ranks_by_wins_descending():
    wins = [10, 50, 30]  # team_0=10, team_1=50, team_2=30
    order_dict = {seed: pick for seed, pick in zip(range(14), range(1, 15))}

    standings, standings_dict = build_standings(wins, order_dict)

    assert [s.agent_name for s in standings] == ["team_1", "team_2", "team_0"]
    assert [s.position for s in standings] == [1, 2, 3]


def test_build_standings_marks_top_16_as_playoffs():
    wins = list(range(30, 0, -1))  # 30 teams, strictly descending win totals
    order_dict = {seed: pick for seed, pick in zip(range(14), range(1, 15))}

    standings, standings_dict = build_standings(wins, order_dict)

    playoff_flags = [s.made_playoffs for s in standings]
    assert playoff_flags[:PLAYOFF_CUTOFF] == [True] * PLAYOFF_CUTOFF
    assert playoff_flags[PLAYOFF_CUTOFF:] == [False] * (30 - PLAYOFF_CUTOFF)


def test_build_standings_playoff_draft_position_mirrors_reverse_standing():
    """Playoff teams (positions 1-16) get draft positions 30 down to 15,
    i.e. the best record picks last."""
    wins = list(range(30, 0, -1))
    order_dict = {seed: pick for seed, pick in zip(range(14), range(1, 15))}

    standings, standings_dict = build_standings(wins, order_dict)

    playoff_entries = [s for s in standings if s.made_playoffs]
    assert playoff_entries[0].draft_position == 30  # #1 seed picks dead last
    assert playoff_entries[-1].draft_position == 15  # #16 seed picks first among playoff teams


def test_build_standings_non_playoff_draft_position_comes_from_lottery():
    """Non-playoff teams' draft positions should come directly from the
    lottery's order_dict via their seed (seed 0 = worst record)."""
    wins = list(range(30, 0, -1))
    # Rig the lottery so seed 0 (worst team, position 30) wins the #1 pick
    order_dict = {0: 1}
    order_dict.update({seed: pick for seed, pick in zip(range(1, 14), range(2, 15))})

    standings, standings_dict = build_standings(wins, order_dict)

    worst_team_entry = next(s for s in standings if s.position == 30)
    assert worst_team_entry.draft_position == 1


def test_build_standings_produces_a_valid_full_draft_order_permutation():
    """draft_position values across all 30 teams should form a permutation
    of 1..30 -- no duplicates, no gaps, since every roster slot needs to
    be assigned in the following rookie draft."""
    wins = list(range(30, 0, -1))
    order_dict = {seed: pick for seed, pick in zip(range(14), range(1, 15))}

    standings, standings_dict = build_standings(wins, order_dict)

    draft_positions = sorted(s.draft_position for s in standings)
    assert draft_positions == list(range(1, 31))


def test_apply_standings_mutates_rewards_and_returns_ordered_draft_list():
    # NOTE: build_standings hardcodes "31 - position" and a top-16 playoff
    # cutoff, both of which assume a full 30-team league -- a 3-team toy
    # example breaks this the same way it would break the original code.
    # See test_season_runner.py module docstring / conversation notes.
    wins = list(range(30, 0, -1))
    order_dict = {seed: pick for seed, pick in zip(range(14), range(1, 15))}
    standings, standings_dict = build_standings(wins, order_dict)
    rewards = {}

    full_draft_order = apply_standings(standings, rewards)

    assert len(rewards) == 30
    assert len(full_draft_order) == 30
    assert set(full_draft_order) == {f"team_{i}" for i in range(30)}


def test_update_win_records_handles_zero_games_played_without_dividing_by_zero():
    state = LeagueState(
        players=np.zeros((1, 5)),
        teams={"team_0": np.zeros(1)},
        team_salaries={"team_0": 0.0},
        team_win_pct={"team_0": 0.5},
        team_standing={"team_0": 15},
        team_has_history={"team_0": 0.0},
    )

    update_win_records(state, wins=[0], games_played=[0])

    assert state.team_win_pct["team_0"] == 0.5  # falls back, doesn't crash
    assert state.team_has_history["team_0"] == 1.0


def test_observe_correct_position():
    wins = list(range(30, 0, -1))
    order_dict = {seed: pick for seed, pick in zip(range(14), range(1, 15))}
    standings, standings_dict = build_standings(wins, order_dict)
    
    assert standings_dict["team_0"].position == 1
    assert standings_dict["team_29"].position == 30