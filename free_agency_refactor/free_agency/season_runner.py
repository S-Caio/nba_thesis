"""
Orchestrates one season. Split into stages so each piece can be tested and
reasoned about independently:

  1. run_game_simulation   -- thin wrapper around season_sim.play_season
  2. update_win_records    -- mutates state.team_win_pct / team_has_history
  3. build_standings       -- PURE: wins -> ranked standings + rewards +
                               draft order. No printing, no state mutation,
                               no randomness (the lottery draw itself lives
                               in season_sim.draft_lottery and is passed in
                               as order_dict). This is the function you
                               actually want unit tests on.
  4. print_standings       -- all I/O, isolated so it never needs testing
  5. simulate_and_reward_season -- thin orchestrator, same public signature
                                    as before so env.py doesn't change.

`reward_func` is your existing reward-shaping function; plug it in from
wherever you already keep it.
"""
from dataclasses import dataclass

import numpy as np

from .constants import LeagueConfig
from .state import LeagueState
from .season_sim import play_season, draft_lottery, seed_position
from .reward import reward_func  # your existing reward-shaping function

PLAYOFF_CUTOFF = 16  # positions 1-16 make the playoffs; matches the original


@dataclass
class StandingEntry:
    agent_name: str
    position: int          # 1 = best record
    wins: int
    reward: float
    draft_position: int    # 1 = first pick next season
    made_playoffs: bool


# --- 1. Game simulation ---------------------------------------------------

def run_game_simulation(state: LeagueState, config: LeagueConfig, game_list):
    """Compute each team's aggregate rating and simulate the season."""
    team_ratings = [np.sum(state.teams[f"team_{i}"]) for i in range(config.n_teams)]
    return play_season(team_ratings, game_list, noise_scale=config.season_noise_scale)


# --- 2. Win record bookkeeping --------------------------------------------

def update_win_records(state: LeagueState, wins, games_played) -> None:
    """Mutates state.team_win_pct / team_has_history from raw wins/games_played."""
    for team_idx, total_wins in enumerate(wins):
        agent_name = f"team_{team_idx}"
        team_games = games_played[team_idx]
        state.team_win_pct[agent_name] = total_wins / team_games if team_games > 0 else 0.5
        state.team_has_history[agent_name] = 1.0


# --- 3. Standings / rewards / draft order (pure, testable) ---------------

def build_standings(wins, order_dict: dict[int, int]) -> tuple[list[StandingEntry], dict[str, int]]:
    """
    Pure function: given each team's win total and a lottery result
    (seed -> pick number), returns the full ranked standings with rewards
    and next season's draft position already computed. Takes no state,
    does no I/O, and is deterministic given its inputs -- the ideal
    target for unit tests.
    """
    ranked_teams = sorted(enumerate(wins), key=lambda item: item[1], reverse=True)
    standings = []
    standings_dict = {}

    for position, (team_idx, total_wins) in enumerate(ranked_teams, start=1):
        agent_name = f"team_{team_idx}"
        made_playoffs = position <= PLAYOFF_CUTOFF

        reward_val = reward_func(position)
        if made_playoffs:
            # reward_val = reward_func(position)
            draft_position = 31 - position
        else:
            draft_seed = seed_position(position)
            draft_position = order_dict[draft_seed]
            # reward_val = reward_func(draft_position, k=0.4)

        entry = StandingEntry(
            agent_name=agent_name,
            position=position,
            wins=int(total_wins),
            reward=reward_val,
            draft_position=draft_position,
            made_playoffs=made_playoffs,
        )

        standings.append(entry)

        standings_dict[agent_name] = int(position)

    return standings, standings_dict


def apply_standings(standings: list[StandingEntry], rewards: dict[str, float]) -> list[str]:
    """Mutates `rewards` in place (matching env's self.rewards contract)
    and returns the full_draft_order derived from the standings."""
    full_draft_order = [None] * len(standings)
    for entry in standings:
        rewards[entry.agent_name] = entry.reward
        full_draft_order[entry.draft_position - 1] = entry.agent_name
    return full_draft_order


# --- 4. Printing (all I/O, isolated) --------------------------------------

def print_standings(standings: list[StandingEntry], agent_name_mapping: dict[str, str]) -> None:
    print("\n" + "=" * 55)
    print(f"{'RANK':<6} | {'TEAM NAME':<25} | {'WINS':<5} | {'REWARD':<8}")
    print("=" * 55)

    for entry in standings:
        real_name = agent_name_mapping[entry.agent_name]
        playoff_marker = "⭐" if entry.made_playoffs else "  "
        print(f"{entry.position:<2} {playoff_marker} | {real_name:<25} | "
              f"{entry.wins:<5} | {entry.reward:.4f}")
        if entry.position == PLAYOFF_CUTOFF:
            print("-" * 55)

    print("=" * 55 + "\n")


# --- 5. Orchestrator -------------------------------------------------------

def simulate_and_reward_season(state: LeagueState, config: LeagueConfig,
                                game_list, agent_name_mapping: dict[str, str],
                                rewards: dict[str, float]) -> tuple[tuple[str], dict[str, int]]:
    """Same public signature as before -- env.py doesn't need to change."""
    wins, games_played = run_game_simulation(state, config, game_list)
    update_win_records(state, wins, games_played)

    _, order_dict = draft_lottery()
    standings, standings_dict = build_standings(wins, order_dict)

    # print_standings(standings, agent_name_mapping)
    return apply_standings(standings, rewards), standings_dict
