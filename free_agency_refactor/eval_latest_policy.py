#%%
import pandas as pd
import numpy as np
from plotnine import *
import os

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env


from free_agency.env import FreeAgencyEnv
from free_agency.constants import LeagueConfig
from train import FreeAgencyMaskedModel 
import time

#%%
ModelCatalog.register_custom_model("free_agency_masked_model", FreeAgencyMaskedModel)

def env_creator(config):
    return PettingZooEnv(FreeAgencyEnv())

def evaluate_and_log_policy(algo, iteration, csv_path="evaluation_win_pct.csv", n_seasons=10, n_trajectories=5):
    """
    Runs multiple standalone evaluation rollouts (trajectories), extracts team
    win percentages, computes league competitive parity, and appends the
    records to a CSV.
    """
    from free_agency.constants import LeagueConfig  # Adjust path to match your layout

    records = []  # accumulate across ALL trajectories

    for traj in range(n_trajectories):
        start_time = time.time()
        print(f"Running trajectory {traj}")
        # 1. Initialize evaluation environment
        eval_config = LeagueConfig()
        eval_config.n_seasons = n_seasons
        eval_env = FreeAgencyEnv(config=eval_config)
        eval_env.reset()

        # Per-trajectory tracker
        last_season = 0

        # 2. Sequential environment rollout loop
        for agent in eval_env.agent_iter():
            obs, reward, termination, truncation, info = eval_env.last()

            if termination or truncation:
                action = None
            else:
                action = algo.compute_single_action(
                    obs,
                    policy_id="shared_policy",
                    explore=False  # Deterministic behavior for objective tracking
                )

            eval_env.step(action)

            # 3. Intercept seasonal boundaries
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
                    "league_std_dev": np.std(list(season_win_p_dict.values()))
                }
                for team_id, win_pct in season_win_p_dict.items():
                    row[team_id] = win_pct

                records.append(row)
                last_season = eval_env.season

        end_time = time.time()
        print(f"Time taken for trajectory {traj}: {end_time - start_time}")

    # 4. Persistence layer using Pandas — happens once, after all trajectories
    df_new = pd.DataFrame(records)

    if not os.path.exists(csv_path):
        df_new.to_csv(csv_path, index=False)
        print(f" Created new tracking log file: {csv_path}")
    else:
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
        print(f" Appended {len(records)} evaluation records to {csv_path}")

def inspect_trained_policy(checkpoint_path):
    checkpoint_path = checkpoint_path = os.path.abspath(checkpoint_path)

    if not ray.is_initialized():
        # Keep things strictly pinned to 1 CPU core for local inference evaluation
        ray.init(num_cpus=1, num_gpus=1)
        
    print(f"\n📂 Rebuilding configuration shell and restoring weights from: {checkpoint_path}")
    
    # Register the environment for our local algorithm instance
    register_env("free_agency_v1", env_creator)
    sample_env = PettingZooEnv(FreeAgencyEnv())
    obs_space = sample_env.observation_space["team_0"]
    act_space = sample_env.action_space["team_0"]

    # 2. Build a duplicate structural config, but lock down worker counts to 0
    config_obj = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("free_agency_v1")
        .framework("torch")
        .resources(num_gpus=0)  # Evaluation runs incredibly fast on CPU
        .env_runners(num_env_runners=0)  # Forces evaluation inside the local main process
        # .rollouts(num_rollout_workers=0)  # Redundant safety flag for older versions
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .training(
            model={
                "custom_model": "free_agency_masked_model",
            }
        )
        .experimental(_disable_preprocessor_api=True)
    )
    
    # Construct the bare model shell and restore the trained weights into it
    algo = config_obj.build_algo()
    algo.restore(checkpoint_path)
    
    # 3. Spin up an evaluation environment instance
    config = LeagueConfig()
    env = FreeAgencyEnv(config=config)
    env.reset()
    
    print("\n==========================================================================")
    print(" 🎬 Collecting many trajectories")
    print("==========================================================================\n")

    evaluate_and_log_policy(algo, 0, csv_path="multiple_trajectories.csv", n_trajectories=40)
            
    ray.shutdown()


if __name__ == "__main__":
    FINAL_CHECKPOINT = "./rllib_checkpoints/periodic"
    inspect_trained_policy(FINAL_CHECKPOINT)


    traj = pd.read_csv("multiple_trajectories.csv")

    traj = traj.melt(
        id_vars=["trajectory", "iteration", "evaluation_season"]
        )

    traj = traj[traj["variable"] != "league_std_dev"]

    from scipy.stats import wasserstein_distance

    nba_win_pct = pd.read_csv("../initial_code/current_system_win_pct_series.csv")

    # Real distribution
    real = nba_win_pct["WinPCT"].to_numpy()

    # Compute Wasserstein distance for each iteration/season
    wasserstein_df = (
        traj
        .groupby(["trajectory"])
        .agg(
            wasserstein=(
                "value",
                lambda x: wasserstein_distance(x.to_numpy(), real)
            )
        )
        .reset_index()
    )

    wasserstein_df
    p = (
        ggplot(wasserstein_df,
            aes(
                x = "wasserstein"
            ))
        + geom_density()
    )

    p.savefig("WASSERSTEIN_PLOT.pdf")

# (
#     ggplot(
#         wasserstein_df,
#         aes(
#             x="iteration",
#             y="wasserstein"
#             )
#     )
#     + geom_line()
#     + geom_point()
#     + labs(
#         x="Iteration",
#         y="Wasserstein distance"
#     )
#     + theme_bw()
# )