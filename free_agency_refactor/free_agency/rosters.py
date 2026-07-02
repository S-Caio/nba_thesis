"""
Roster bookkeeping. These functions derive `teams` / `team_salaries` from
the source of truth (the TEAM column in `players`), and report on current
occupancy. Nothing here decides *who* gets signed -- that's contracts.py.
"""
import numpy as np

from .constants import LeagueConfig, RATING, TEAM, SALARY
from .state import LeagueState


def get_team_counts(state: LeagueState, config: LeagueConfig) -> dict[str, int]:
    counts = {f"team_{i}": 0 for i in range(config.n_teams)}
    owned_mask = state.players[:, TEAM] > 0
    for marker in state.players[owned_mask, TEAM].astype(int):
        counts[f"team_{marker - 1}"] += 1
    return counts


def rebuild_rosters(state: LeagueState, config: LeagueConfig) -> None:
    """Recompute `teams` and `team_salaries` from scratch based on each
    player's TEAM ownership marker. Call this after any bulk player-array
    mutation (aging, retirement, rookie draft) to keep views in sync."""
    state.teams = {agent: np.zeros(config.players_per_team) for agent in state.teams}
    state.team_salaries = {agent: 0.0 for agent in state.team_salaries}

    for player in state.players:
        team_marker = int(player[TEAM])
        if team_marker <= 0:
            continue
        agent = f"team_{team_marker - 1}"
        team_vector = state.teams[agent]
        empty_slots = np.where(team_vector == 0.0)[0]
        if len(empty_slots) > 0:
            state.teams[agent][empty_slots[0]] = player[RATING]
            state.team_salaries[agent] += player[SALARY]


def print_team_rosters(state: LeagueState, config: LeagueConfig,
                        agent_name_mapping: dict[str, str]) -> None:
    print("\n" + "═" * 75)
    print(f"{'TEAM NAME':<25} | {'ROSTER RATINGS (Top 10 Slots)':<35} | {'TOTAL STRENGTH':<12}")
    print("═" * 75)

    for agent, team_vector in state.teams.items():
        real_name = agent_name_mapping[agent]
        active_players = team_vector[team_vector > 0.0]
        total_strength = np.sum(team_vector)

        if len(active_players) > 0:
            ratings_str = ", ".join(f"{r:.2f}" for r in active_players)
            if len(ratings_str) > 33:
                ratings_str = ratings_str[:30] + "..."
            roster_display = f"[{ratings_str}] ({len(active_players)}/{config.players_per_team})"
        else:
            roster_display = f"Empty (0/{config.players_per_team})"

        print(f"{real_name:<25} | {roster_display:<35} | {total_strength:<12.2f}")

    print("═" * 75 + "\n")
