
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time

from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN

from free_agency.env_parallel import FreeAgencyEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import psutil, os

class FreeAgencyMaskedModel(TorchModelV2, nn.Module):
    """
    Expects the Dict observation space defined in FreeAgencyEnv:
        action_mask              : (n_actions,)
        free_agents               : (n_free_agents, N_PLAYER_COLS)
        my_team                  : (players_per_team,)
        my_team_rating            : (1,)
        my_team_avg_age            : (1,)
        n_players_team              : (1,)
        win_pct                    : (1,)
        season                      : (1,)
        team_salary                  : (1,)
        standing                      : (1,)
        has_history                    : (1,)
        relative_team_strength          : (1,)   # z-score of team strength vs league
        n_players_team_relative           : (1,)   # z-score of roster headcount vs league
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.orig_space = getattr(obs_space, "original_space", obs_space)

        self.n_free_agents, self.n_player_cols = self.orig_space["free_agents"].shape
        self.players_per_team = self.orig_space["my_team"].shape[0]
        self.history_window = self.orig_space["win_pct_history"].shape[0]
        self.cap_horizon = self.orig_space["cap_projection"].shape[0]

        player_embed_dim = 32
        conv_hidden = model_config.get("custom_model_config", {}).get("conv_hidden", 64)

        # Positional encoder over the ratings-sorted free-agent list.
        self.player_conv = nn.Sequential(
            nn.Conv1d(self.n_player_cols, conv_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(conv_hidden, player_embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )
        player_feat_dim = player_embed_dim * self.n_free_agents

        self.market_encoder = nn.Sequential(
            nn.Linear(self.n_free_agents * player_embed_dim, 256),
            nn.LeakyReLU(),
        )

        # my_team_rating, win_pct, team_salary, standing, has_history,
        # relative_team_strength, n_players_team_relative,
        # my_team_avg_age, n_players_team
        scalar_dim = 9
        history_input_dim = self.history_window * 3
        team_raw_dim = self.players_per_team + scalar_dim + history_input_dim + self.cap_horizon
        self.team_norm = nn.LayerNorm(team_raw_dim)
        self.team_encoder = nn.Sequential(
            nn.Linear(team_raw_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
        )

        combined_dim = 256 + 128 + 32
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )

        self.logits_layer = nn.Linear(128, num_outputs)
        self.value_layer = nn.Linear(128, 1)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        if not isinstance(obs, dict):
            obs = restore_original_dimensions(obs, self.obs_space, "torch")

        action_mask = obs["action_mask"].float()
        free_agents = obs["free_agents"].float()   # (B, n_free_agents, n_cols)
        my_team = obs["my_team"].float()
        my_team_rating = obs["my_team_rating"].float()
        my_team_avg_age = obs["my_team_avg_age"].float()
        n_players_team = obs["n_players_team"].float()
        win_pct = obs["win_pct"].float()
        team_salary = obs["team_salary"].float()
        standing = obs["standing"].float()
        has_history = obs["has_history"].float()
        relative_team_strength = obs["relative_team_strength"].float()
        n_players_team_relative = obs["n_players_team_relative"].float()
        history_mask = obs["history_mask"].float()
        win_pct_history = obs["win_pct_history"].float() * history_mask
        rating_history = obs["rating_history"].float() * history_mask
        cap_projection = obs["cap_projection"].float()

        # Conv1d wants channel-first: (B, C=n_cols, L=n_free_agents)
        x = free_agents.permute(0, 2, 1)
        x = self.player_conv(x)
        market_summary = torch.mean(x, dim=2)
        x_flat = x.reshape(x.shape[0], -1)
        market_feat = self.market_encoder(x_flat)

        history_vars = torch.cat([win_pct_history, rating_history, history_mask], dim=-1)
        scalars = torch.cat([
            my_team_rating, win_pct, team_salary, standing, has_history,
            relative_team_strength, n_players_team_relative,
            my_team_avg_age, n_players_team,
        ], dim=-1)

        raw_team_context = torch.cat([my_team, scalars, history_vars, cap_projection], dim=-1)
        normed_team_context = self.team_norm(raw_team_context)
        team_feat = self.team_encoder(normed_team_context)

        combined = torch.cat([market_feat, market_summary, team_feat], dim=-1)
        trunk_out = self.trunk(combined)

        logits = self.logits_layer(trunk_out)
        self._value_out = self.value_layer(trunk_out).squeeze(-1)

        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        return masked_logits, state

    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return self._value_out

ModelCatalog.register_custom_model("free_agency_masked_model", FreeAgencyMaskedModel)



def evaluate_and_log_policy(algo, iteration, csv_path="evaluation_win_pct.csv", n_seasons=10, n_trajectories=5):
    from free_agency.constants import LeagueConfig

    records = []

    for traj in range(n_trajectories):
        eval_config = LeagueConfig()
        eval_config.n_seasons = n_seasons
        eval_env = FreeAgencyEnv(config=eval_config)  # raw parallel env, not wrapped
        obs, infos = eval_env.reset()



        last_season = 0

        # ParallelEnv convention: eval_env.agents is empty once the episode ends
        while eval_env.season <= n_seasons and eval_env.agents:
            actions = {
                agent_id: algo.compute_single_action(
                    agent_obs,
                    policy_id="shared_policy",
                    explore=False,  # deterministic for eval
                )
                for agent_id, agent_obs in obs.items()
            }

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
    else:
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
        print(f" Appended {len(records)} evaluation records to {csv_path}")



# ---------------------------------------------------------------------------
# Usage sketch: wiring FreeAgencyEnv (a PettingZoo AECEnv) + this model into
# RLlib's PPO. Adjust import paths for your project layout.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ITER = 10_000

    def env_creator(config):
        # PettingZooEnv wraps an AECEnv for RLlib's multi-agent API.
        return ParallelPettingZooEnv(FreeAgencyEnv())  # noqa: F821 (import above)

    register_env("free_agency_v1", env_creator)

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1024**3)

    sample_env = ParallelPettingZooEnv(FreeAgencyEnv())  # noqa: F821
    obs_space = sample_env.observation_space["team_0"]
    act_space = sample_env.action_space["team_0"]

    # print(f"OBS SPACE: {obs_space}")

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("free_agency_v1")
        .framework("torch")
        .resources(num_gpus=1)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .training(
                train_batch_size=512,
                # minibatch_size=32,
                # num_epochs=1,

                    model={
                        "custom_model": "free_agency_masked_model",
                        # example: "custom_model_config": {"conv_hidden": 64},
                    }
                )
        .experimental(_disable_preprocessor_api=True)
        .env_runners(
                # num_env_runners=0,       # Uses 2 CPU cores for parallel environment simulation
        #         num_cpus_per_env_runner=1, # 1 CPU core per worker
        #         sample_timeout_s=300.0,  # Prevents timeouts during heavy season simulations
                )
        
    )

    cfg = config.to_dict()

    print(cfg["train_batch_size"])
    print(cfg["rollout_fragment_length"])
    print(cfg.get("minibatch_size"))

    algo = config.build_algo()
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1024**3)

    log_file = "free_agency_env_win_pct.csv"
    print("Training has started!")
    start_time = time.time()

    for i in range(ITER):
        print(f"Training in iteration {i}")
        start_time_iter = time.time()
        print("Calling algo.train()")
        result = algo.train()
        print("Returned from algo.train()")

        print("Result:")
        print(result)

        # Safely pull env_runners dict if present, falling back to top-level result
        metrics_source = result.get("env_runners", result)

        # Extract episode metrics (populates once episodes start completing)
        ep_reward_mean = metrics_source.get("episode_reward_mean", np.nan)
        ep_len_mean = metrics_source.get("episode_len_mean", np.nan)
        
        # Extract completed episode count across versions
        ep_count = metrics_source.get("num_episodes", metrics_source.get("episodes_this_iter", 0))

        # Extract multi-agent policy specific reward
        policy_rewards = metrics_source.get("policy_reward_mean", result.get("policy_reward_mean", {}))
        shared_policy_reward = policy_rewards.get("shared_policy", np.nan)

        print(
            f"iter {i}: episode_reward_mean={ep_reward_mean:.2f} "
            f"(over {ep_count} completed episodes, avg len: {ep_len_mean}) | "
            f"shared_policy_reward={shared_policy_reward:.2f}"
        )
        # # Extract metrics safely from the nested env_runners dictionary
        # env_runners = result.get("env_runners", {})
        # ep_reward_mean = env_runners.get("episode_reward_mean")
        # ep_len_mean = env_runners.get("episode_len_mean")
        # ep_count = env_runners.get("num_episodes")

        # print(f"iter {i}: episode_reward_mean={ep_reward_mean} (over {ep_count} episodes, avg len: {ep_len_mean})")
        # print(f"Result: {result}")
        
        # print({
        #     "env_steps": result.get("num_env_steps_sampled"),
        #     "agent_steps": result.get("num_agent_steps_sampled"),
        # })

        if i % 25 == 0:
            print(f"Running 10-season evaluation episode...")
            eval_metrics = evaluate_and_log_policy(algo, n_seasons=10, csv_path = log_file, iteration = i)

        if i % 100 == 0:
            periodic_path = algo.save(checkpoint_dir="./rllib_checkpoints/periodic")
            print(f"Periodic checkpoint saved at: {periodic_path}")

        end_time_iter = time.time()
        print(f"Time for iteration: {end_time_iter - start_time_iter}")

        process = psutil.Process(os.getpid())
        print("Memory info: ", process.memory_info().rss / 1024**3)
        
    final_path = algo.save(checkpoint_dir="./rllib_checkpoints/final")
    print(f"\n Training complete! Final model saved to: {final_path}")
    end_time = time.time()
    print(f"Training for {ITER} iterations took {end_time - start_time}")

