import numpy as np

from free_agency.constants import LeagueConfig, RATING, TEAM, SALARY
from free_agency.state import LeagueState
from free_agency.rosters import get_team_counts, rebuild_rosters


def make_state():
    players = np.array([
        [4.0, 1, 25, 2, 10],  # owned by team_0
        [3.0, 1, 26, 1, 8],   # owned by team_0
        [2.0, 2, 27, 3, 6],   # owned by team_1
        [1.0, 0, 28, 0, 0],   # free agent
    ])
    teams = {"team_0": np.zeros(2), "team_1": np.zeros(2)}
    team_salaries = {"team_0": 0.0, "team_1": 0.0}
    return LeagueState(players, teams, team_salaries, {"team_0": 0.5, "team_1": 0.5},
                        {"team_0": 0.0, "team_1": 0.0})


def make_config():
    return LeagueConfig(n_teams=2, players_per_team=2)


def test_get_team_counts_counts_only_owned_players():
    state = make_state()
    config = make_config()

    counts = get_team_counts(state, config)

    assert counts["team_0"] == 2
    assert counts["team_1"] == 1


def test_rebuild_rosters_repopulates_ratings_and_salaries_from_ownership():
    state = make_state()
    config = make_config()
    # Wipe teams/salaries to confirm rebuild derives them purely from `players`
    state.teams = {"team_0": np.zeros(2), "team_1": np.zeros(2)}
    state.team_salaries = {"team_0": 0.0, "team_1": 0.0}

    rebuild_rosters(state, config)

    assert sorted(state.teams["team_0"].tolist()) == [3.0, 4.0]
    assert state.team_salaries["team_0"] == 18.0  # 10 + 8
    assert sorted(state.teams["team_1"].tolist()) == [0.0, 2.0]
    assert state.team_salaries["team_1"] == 6.0


def test_rebuild_rosters_drops_overflow_players_beyond_roster_size():
    """If more players are marked as owned by a team than it has slots for,
    rebuild should fill available slots and silently ignore the rest,
    rather than raising an IndexError."""
    players = np.array([
        [4.0, 1, 25, 2, 10],
        [3.0, 1, 26, 1, 8],
        [2.0, 1, 27, 1, 6],  # third player for a 2-slot roster
    ])
    state = LeagueState(players, {"team_0": np.zeros(2)}, {"team_0": 0.0},
                         {"team_0": 0.5}, {"team_0": 0.0})
    config = LeagueConfig(n_teams=1, players_per_team=2)

    rebuild_rosters(state, config)  # should not raise

    assert len(state.teams["team_0"]) == 2
