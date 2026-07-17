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


def make_test_state() -> LeagueState:
    """
    Generates a minimal 2-team, 3-player LeagueState for unit testing.
    - team_0: 1/2 roster slots filled, but at the 100.0 salary cap.
    - team_1: 2/2 roster slots filled (full roster), but under the cap (25.0).
    """
    # 1. Global Player Pool (3 players total, 5 columns each)
    # Columns: [RATING, TEAM_IDX, AGE, CONTRACT_LEN, SALARY]
    players_matrix = np.array([
        [85.0,  1.0, 26.0, 3.0, 95.0],  # Player 0: Signed to team_0, maxes cap
        [70.0,  2.0, 22.0, 1.0,  10.0],  # Player 1: Signed to team_1
        [72.0,  2.0, 24.0, 2.0,  15.0],  # Player 2: Signed to team_1 (Roster full)
        [10.0, 0.0, 21.0, 0.0, 0.0]
    ], dtype=np.float32)

    # 2. Team Roster Arrays (Tracks individual player ratings in active slots)
    # Assumes a max roster size config of 2 players per team
    teams_rosters = {
        "team_0": np.array([85.0, 0.0], dtype=np.float32),   # Slot 2 open, but cap-blocked
        "team_1": np.array([70.0, 72.0], dtype=np.float32),  # No slots open
    }

    # 3. Financials & Metadata
    team_salaries = {"team_0": 95.0, "team_1": 25.0}
    team_win_pct  = {"team_0": 0.5,   "team_1": 0.5}
    team_has_history = {"team_0": 1.0, "team_1": 1.0}

    return LeagueState(
        players=players_matrix,
        teams=teams_rosters,
        team_salaries=team_salaries,
        team_win_pct=team_win_pct,
        team_has_history=team_has_history
    )


def make_test_config() -> LeagueConfig:
    """
    Generates a matching LeagueConfig for the minimal 2-team test state.
    Configures structural boundaries to match the mock data precisely.
    """
    config = LeagueConfig()
    
    # Structural dimensions matching the test state
    config.n_teams = 2
    config.players_per_team = 2
    
    # Financial parameters for validation logic
    config.salary_cap = 100.0
    config.salary_ranges = [5.0, 10.0, 15.0, 50.0, 100.0]
    config.contract_lengths = [1, 2, 3, 4]
    
    # Simulation defaults
    config.n_seasons = 3
    config.n_new_entrants_per_season = 1
    config.season_noise_scale = 0.0
    
    return config


def get_action_map(config: LeagueConfig) -> np.ndarray:
    """
    Generates a flattened array of all possible action combinations.
    Rows match the single scalar action index.
    Columns map to: [player_idx, salary_idx, contract_length_idx]
    """
    n_players = config.n_players
    n_salaries = len(config.salary_ranges)
    n_lengths = len(config.contract_lengths)
    
    # Create an exhaustive grid of all combinations
    grid = np.mgrid[0:n_players, 0:n_salaries, 0:n_lengths]
    
    # Reshape into an (N, 3) matrix where N = n_players * n_salaries * n_lengths
    action_map = grid.reshape(3, -1).T
    return action_map

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

    action_map = get_action_map(config)
    # print("Action Map:")
    # print(action_map)
    action_salaries = action_map[:, 1] # Extract salaries
    # print(action_salaries) 
    salary_values = np.array([config.salary_ranges[i] for i in action_salaries])
    # print(salary_values)

    salary_mask = np.where(salary + salary_values <= config.salary_cap, 1, 0)
    
    num_players = len(team_vec[team_vec > 0.0])
    print(f"Num players: {num_players}")
    if num_players + 1 <= config.players_per_team:
        num_players_mask = np.ones_like(salary_mask)
    else:
        num_players_mask = np.zeros_like(salary_mask)

    player_id_actions = action_map[:, 0].astype(int)
    available_players_mask = np.where(state.players[player_id_actions, TEAM] == 0, 1, 0)

    mask_all = num_players_mask & salary_mask & available_players_mask

    return mask_all



def test_action_mask_creation():
    """
    One team that is at the salary cap, another that already has 10 players
    """

    state = make_test_state()
    config = make_test_config()

    for team in ["team_0", "team_1"]:
        print(team)
        make_action_mask(state, config, team)

    

    


     
