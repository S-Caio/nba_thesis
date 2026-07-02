import numpy as np
import pytest

from free_agency.season_sim import generate_exact_nba_schedule, play_season, draft_lottery, seed_position


def test_schedule_generates_expected_total_game_count():
    """Sanity check on the combinatorics: with the standard 2/3/4-game
    matchup rules for a 30-team, 15/15 conference, 5-team division league,
    the total game count should match the real 82-game-season structure."""
    schedule = generate_exact_nba_schedule(n_teams=30)
    # Each team plays 82 games/season -> 30*82/2 total games (each game counted once)
    assert len(schedule) == 30 * 82 // 2


def test_schedule_covers_every_team_pair_at_least_once():
    schedule = generate_exact_nba_schedule(n_teams=30)
    pairs_seen = {tuple(sorted(g)) for g in schedule}
    expected_pairs = {(i, j) for i in range(30) for j in range(i + 1, 30)}
    assert pairs_seen == expected_pairs


def test_play_season_conserves_total_games():
    """Every simulated game should award exactly one win and increment
    games_played for both participating teams -- no games lost or double
    counted regardless of the random noise."""
    schedule = [(0, 1), (1, 2), (0, 2)] * 10
    wins, games_played = play_season([5.0, 3.0, 1.0], schedule, noise_scale=10)

    assert wins.sum() == len(schedule)
    assert games_played.sum() == 2 * len(schedule)


def test_play_season_strongly_favors_much_better_team_with_low_noise():
    schedule = [(0, 1)] * 500
    wins, _ = play_season([100.0, 0.0], schedule, noise_scale=1)  # huge rating gap, low noise
    assert wins[0] > wins[1]
    assert wins[0] / 500 > 0.9


def test_draft_lottery_returns_a_permutation_of_all_14_seeds():
    draft_order, order_dict = draft_lottery()
    assert sorted(draft_order) == list(range(14))
    assert sorted(order_dict.keys()) == list(range(14))
    assert sorted(order_dict.values()) == list(range(1, 15))


def test_draft_lottery_picks_5_through_14_follow_seed_order():
    """Only the top 4 picks are subject to lottery reordering; picks 5-14
    should always go worst-record-first among whichever seeds didn't win
    a top-4 slot."""
    draft_order, _ = draft_lottery()
    assert draft_order[4:] == sorted(draft_order[4:])


def test_draft_lottery_worst_teams_win_first_pick_more_often_than_best_lottery_team():
    """Statistical property, not exact odds: seed 0 (worst record) should
    win the #1 pick meaningfully more often than seed 13 (best of the
    lottery teams) over many trials."""
    np.random.seed(0)
    first_pick_counts = {0: 0, 13: 0}
    trials = 2000
    for _ in range(trials):
        draft_order, _ = draft_lottery()
        if draft_order[0] in first_pick_counts:
            first_pick_counts[draft_order[0]] += 1

    assert first_pick_counts[0] > first_pick_counts[13] * 3


def test_seed_position_is_inverse_relationship():
    assert seed_position(1) == 29
    assert seed_position(30) == 0
