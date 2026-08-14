import os

import numpy as np
import pandas as pd
import torch

from ray.rllib.core.columns import Columns
from free_agency.env_parallel import FreeAgencyEnv


def _batch_obs_for_module(obs_dict, device):
    """
    obs_dict: {agent_id: {obs_key: np.ndarray, ...}, ...} -- one Dict-space
    observation per agent, all agents sharing the same policy/module.

    Returns (agent_ids, batched_obs) where batched_obs is a dict matching the
    module's observation space, with a leading batch dim of len(agent_ids).
    """
    agent_ids = list(obs_dict.keys())
    obs_keys = obs_dict[agent_ids[0]].keys()

    batched_obs = {}
    for key in obs_keys:
        stacked = np.stack([obs_dict[aid][key] for aid in agent_ids], axis=0)
        batched_obs[key] = torch.as_tensor(stacked, device=device)

    return agent_ids, batched_obs


@torch.no_grad()
def _compute_greedy_actions(rl_module, obs_dict, device):
    """Greedy (explore=False) actions for every agent in obs_dict, via one
    batched forward pass through the shared RLModule."""
    agent_ids, batched_obs = _batch_obs_for_module(obs_dict, device)

    out = rl_module.forward_inference({Columns.OBS: batched_obs})
    masked_logits = out[Columns.ACTION_DIST_INPUTS]

    # Discrete action space -> deterministic/greedy == argmax of the
    # (already action-masked) logits.
    greedy_actions = torch.argmax(masked_logits, dim=-1).cpu().numpy()

    return {aid: int(a) for aid, a in zip(agent_ids, greedy_actions)}


def evaluate_and_log_policy(
    algo, iteration, csv_path="evaluation_win_pct.csv", n_seasons=10, n_trajectories=5
):
    from free_agency.constants import LeagueConfig

    # Pull the RLModule once (it's the same shared module for every agent
    # and every trajectory below -- no need to re-fetch it in the loop).
    rl_module = algo.get_module("shared_policy")
    if rl_module is None:
        raise ValueError(
            "algo.get_module('shared_policy') returned None -- check that "
            "'shared_policy' matches the module id used in your "
            "multi_agent(policies=...) / rl_module_spec config."
        )
    rl_module.eval()
    device = next(rl_module.parameters()).device

    records = []

    for traj in range(n_trajectories):
        eval_config = LeagueConfig()
        eval_config.n_seasons = n_seasons
        eval_env = FreeAgencyEnv(config=eval_config)  # raw parallel env, not wrapped
        obs, infos = eval_env.reset()

        last_season = 0

        # ParallelEnv convention: eval_env.agents is empty once the episode ends
        while eval_env.season <= n_seasons and eval_env.agents:
            actions = _compute_greedy_actions(rl_module, obs, device)

            obs, rewards, terminations, truncations, infos = eval_env.step(actions)

            # Season-boundary bookkeeping — unchanged, assuming num_moves/season
            # are still tracked as attributes on the underlying env instance.
            if eval_env.num_moves == 0 and eval_env.season > last_season:
                completed_season = last_season

                season_win_p_dict = {
                    team: float(eval_env.league.team_win_pct[team])
                    for team in eval_env.possible_agents
                }

                row = {
                    "trajectory": traj,
                    "iteration": iteration,
                    "evaluation_season": completed_season,
                    "league_std_dev": np.std(list(season_win_p_dict.values())),
                }
                row.update(season_win_p_dict)
                records.append(row)
                last_season = eval_env.season

    df_new = pd.DataFrame(records)
    if not os.path.exists(csv_path):
        df_new.to_csv(csv_path, index=False)
        print(f" Created new tracking log file: {csv_path}")
    elif iteration == 0:
        df_new.to_csv(csv_path, index=False)
        print(f" Created new tracking log file: {csv_path}")
    else:
        df_new.to_csv(csv_path, mode="a", header=False, index=False)
        print(f" Appended {len(records)} evaluation records to {csv_path}")