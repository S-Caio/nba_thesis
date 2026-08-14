#%%
import numpy as np
import pandas as pd
import os


from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
# Import the correct class to create from scratch using the checkpoint.
from ray.rllib.algorithms.algorithm import Algorithm


import ray

from free_agency.env_parallel import FreeAgencyEnv
from train_parallel import FreeAgencyMaskedModel, evaluate_and_log_policy

#%%
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# from free_agency_env import FreeAgencyEnv  # your module

ITER = 10_000

def env_creator(config):
    # PettingZooEnv wraps an AECEnv for RLlib's multi-agent API.
    return PettingZooEnv(FreeAgencyEnv())  # noqa: F821 (import above)

register_env("free_agency_v1", env_creator)
print("#" * 15)
print("Environment Registration Completed")
print("#" * 15)
if not ray.is_initialized():
    ray.init(
        num_cpus=1,
        num_gpus=0,
        include_dashboard=False,
        ignore_reinit_error=True,
    )
    print(f"Cluster resources: {ray.cluster_resources()}")
    print(f"Available resources: {ray.available_resources()}")
    print("#" * 30)
    print("Initialised ray")
    print("#" * 30)

CHECKPOINT_DIR = os.path.abspath("./rllib_checkpoints/periodic")

sample_env = ParallelPettingZooEnv(FreeAgencyEnv())  # noqa: F821
obs_space = sample_env.observation_space["team_0"]
act_space = sample_env.action_space["team_0"]

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment("free_agency_v1")
    .framework("torch")
    .resources(num_gpus=0)
    .env_runners(
        num_env_runners=1,       # <-- downsized from 2
        num_cpus_per_env_runner=1,
        sample_timeout_s=300.0,
        num_gpus_per_env_runner = 0
    )
    .multi_agent(
        policies={
            "shared_policy": (None, obs_space, act_space, {})
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs:
            "shared_policy",
    )
    .training(
        model={
            "custom_model": "free_agency_masked_model",
        }
    )
    .experimental(
        _disable_preprocessor_api=True
    )
)

algo = config.build_algo()

print("Algorithm constructed")
print("Restoring checkpoint...")

algo.restore(CHECKPOINT_DIR)

print("Checkpoint restored!")
#%%

print(f"We are in iteration {algo.iteration}")
print(algo._counters)
