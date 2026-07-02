import numpy as np

from free_agency.season_runner import (
    build_standings, apply_standings, update_win_records, PLAYOFF_CUTOFF,
)

from free_agency.constants import LeagueConfig, RATING, TEAM, AGE, CONTRACT_LEN, SALARY
from free_agency.state import LeagueState
from free_agency.player_lifecycle import player_update, assign_rookie_draft
from free_agency.rosters import rebuild_rosters
from free_agency.contracts import handle_signing
from free_agency.season_runner import build_standings, apply_standings, print_standings
from free_agency.season_sim import draft_lottery

def make_state(n_players=6, players_per_team=3, agents=("team_0", "team_1")):
    """Minimal deterministic league: 4 players, ratings 4,3,2,1, all free agents."""
    players = np.array([
        [4.0, 1, 25, 2, 10],
        [3.0, 1, 26, 2, 5],
        [2.0, 2, 27, 2, 10],
        [1.0, 2, 28, 2, 5],
        [0.5, 0, 34, 0, 0],
        [0.25, 0, 19, 0, 0]
    ])[:n_players]
    teams = {a: np.zeros(players_per_team) for a in agents}
    team_salaries = {a: 20.0 for a in agents}
    team_win_pct = {a: 0.5 for a in agents}
    team_has_history = {a: 0.0 for a in agents}
    return LeagueState(players, teams, team_salaries, team_win_pct, team_has_history)


def make_config():
    return LeagueConfig(salary_cap=20.0,
                         salary_ranges=np.array([5, 10, 15]),
                         contract_lengths=np.array([1, 2, 3]))

def test_try_invalid_signing():
    state = make_state()
    config = make_config()

    rebuild_rosters(state, config)

    # print(f"PLAYERS: {state.players}")
    # print(f"TEAMS: {state.teams}")
    # print(f"SALARIES: {state.team_salaries}")
    # print(f"SALARIES SUM: {list((state.team_salaries.values()))}")

    handle_signing(state, config, "team_0", (2, 0, 0))

    # print(f"PLAYERS: {state.players}")
    assert state.players[2, TEAM] == 2

    handle_signing(state, config, "team_0", (4, 0, 0))

    # print(f"PLAYERS: {state.players}")
    # print(f"SALARIES: {state.team_salaries}")
    assert state.players[4, TEAM] == 1
    assert state.team_salaries["team_0"] == 20.0



# def make_state(n_players=6, players_per_team=3, agents=("team_0", "team_1")):
#     """Minimal deterministic league: 4 players, ratings 4,3,2,1, all free agents."""
#     players = np.array([
#         [4.0, 1, 25, 2, 10],
#         [3.0, 1, 26, 2, 5],
#         [2.0, 2, 27, 2, 10],
#         [1.0, 2, 28, 2, 5],
#         [0.5, 0, 34, 0, 0],
#         [0.25, 0, 19, 0, 0]
#     ])[:n_players]
#     teams = {a: np.zeros(players_per_team) for a in agents}
#     team_salaries = {a: 20.0 for a in agents}
#     team_win_pct = {a: 0.5 for a in agents}
#     team_has_history = {a: 0.0 for a in agents}
#     return LeagueState(players, teams, team_salaries, team_win_pct, team_has_history)


def make_config():
    return LeagueConfig(salary_cap=100.0,
                         salary_ranges=np.array([5, 10, 15]),
                         contract_lengths=np.array([1, 2, 3]))

def test_best_lottery_players_go_in_draft_order():
    team_names = ["team_0", "team_1"]
    players = np.array([
        [1, 0, 21, 0, 0],
        [5, 0, 21, 0, 0],
    ]
    )

    teams = {a : np.zeros(1) for a in team_names}
    salaries = {a : 0.0 for a in team_names}
    state = LeagueState(players, teams, salaries,
                         {"team_0": 0.5, "team_1": 0.5}, {"team_0": 0.0, "team_1": 0.0})
    config = LeagueConfig(n_teams=2, players_per_team=1, salary_cap=100.0, rookie_salary=5)

    assign_rookie_draft(state, config, full_draft_order=["team_1", "team_0"],
                         rookie_idx=np.array([0, 1]))
    print(f"PLAYERS: \n {state.players}")

    assert state.players[0, TEAM] == 1
    assert state.players[1, TEAM] == 2

    


     
