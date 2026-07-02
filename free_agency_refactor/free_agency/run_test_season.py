"""
Runs the env for exactly one season using random actions. Good for a fast
sanity check: does the draft complete, does the season simulate, do
rewards get assigned, does the env advance correctly, without errors.

Run: python -m free_agency.run_test_season
"""
from free_agency.env import FreeAgencyEnv
from free_agency.constants import LeagueConfig


def run_one_season(seed: int = 0):
    config = LeagueConfig(n_teams=30, players_per_team=10, n_seasons=1)
    env = FreeAgencyEnv(config=config)
    env.reset(seed=seed)

    steps = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()

        env.step(action)
        steps += 1

        if all(env.terminations.values()) or all(env.truncations.values()):
            break

    print(f"\nCompleted {steps} steps.")
    print("Final rewards:", env.rewards)
    print("Final win pct:", env.league.team_win_pct)


if __name__ == "__main__":
    run_one_season()
