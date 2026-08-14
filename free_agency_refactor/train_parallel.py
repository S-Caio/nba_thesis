
import numpy as np
import pandas as pd
import os
import time
from training_utils.nn import FreeAgencyMaskedModel
from training_utils.evaluate_and_log_policy import evaluate_and_log_policy

from ray.rllib.models import ModelCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec


from free_agency.env_parallel import FreeAgencyEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import psutil, os

# Register model
ModelCatalog.register_custom_model("free_agency_masked_model", FreeAgencyMaskedModel)

if __name__ == "__main__":
    ITER = 10_000

    def env_creator(config):
        return ParallelPettingZooEnv(FreeAgencyEnv()) 

    register_env("free_agency_v1", env_creator)

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1024**3)

    sample_env = ParallelPettingZooEnv(FreeAgencyEnv())
    obs_space = sample_env.observation_space["team_0"]
    act_space = sample_env.action_space["team_0"]

    # print(f"OBS SPACE: {obs_space}")

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
        .environment("free_agency_v1")
        .framework("torch")
        .learners(num_learners=1, num_gpus_per_learner=1)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kw: "shared_policy",
        )
        .rl_module(rl_module_spec=RLModuleSpec(module_class=FreeAgencyMaskedModel))
        .training(
            train_batch_size=512,
            minibatch_size=512, 
            num_epochs=5,
        )
        .experimental(_disable_preprocessor_api=True)
        .env_runners(
            num_env_runners=6,
            num_cpus_per_env_runner=1,
            # num_gpus_per_env_runner=0,
        )
    )

    cfg = config.to_dict()

    print(f"Train batch size = {cfg["train_batch_size"]}")
    print(f"Rollout fragment length = {cfg["rollout_fragment_length"]}")
    print(f"Minibatch size: {cfg.get("minibatch_size")}")

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

        
        end_time_iter = time.time()
        print(f"Time for iteration: {end_time_iter - start_time_iter}")
        
        if i % 25 == 0:
            print(f"Running 10-season evaluation episode...")
            eval_metrics = evaluate_and_log_policy(algo, n_seasons=10, csv_path = "free_agency_env_win_pct.csv", iteration = i)

        if i % 100 == 0:
            # periodic_path = algo.save(checkpoint_dir="./rllib_checkpoints/periodic")
            path = os.path.abspath("./rllib_checkpoints/saves")
            periodic_path = algo.save_to_path(path)
            print(f"Periodic checkpoint saved at: {periodic_path}")

        process = psutil.Process(os.getpid())
        print("Memory info: ", process.memory_info().rss / 1024**3)
        
    final_path = algo.save(checkpoint_dir="./rllib_checkpoints/final")
    print(f"\n Training complete! Final model saved to: {final_path}")
    end_time = time.time()
    print(f"Training for {ITER} iterations took {end_time - start_time}")

