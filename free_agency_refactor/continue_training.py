import os
import re
import glob
import time
import psutil
import numpy as np
import torch

from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from free_agency.env_parallel import FreeAgencyEnv
from train_parallel import FreeAgencyMaskedModel, evaluate_and_log_policy



CHECKPOINT_PATH = os.path.abspath(
    "./rllib_checkpoints/periodic"
)

# Register exactly the same model name as before.
ModelCatalog.register_custom_model(
    "free_agency_masked_model",
    FreeAgencyMaskedModel,
)


# ============================================================
# 2. Environment
# ============================================================

def env_creator(config):
    return ParallelPettingZooEnv(
        FreeAgencyEnv()
    )


register_env(
    "free_agency_v1",
    env_creator,
)


# ============================================================
# 3. Find the latest periodic checkpoint
# ============================================================

CHECKPOINT_DIR = "./rllib_checkpoints/periodic"


def find_latest_checkpoint(directory):

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {directory}"
        )

    candidates = []

    # RLlib normally creates directories such as:
    #
    # checkpoint_000100
    # checkpoint_000200
    #
    for path in glob.glob(
        os.path.join(directory, "checkpoint_*")
    ):
        if not os.path.isdir(path):
            continue

        match = re.search(
            r"checkpoint_(\d+)$",
            os.path.basename(path),
        )

        if match:
            iteration = int(match.group(1))
            candidates.append(
                (iteration, path)
            )

    if not candidates:
        raise RuntimeError(
            f"No RLlib checkpoints found in {directory}"
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    return candidates[-1]


# ============================================================
# 4. Build the SAME PPO configuration
# ============================================================

# This should match the configuration used to create
# the original checkpoints.

TARGET_ITER = 10_000

# Number of parallel environment runners.
# Start with 2 or 4 and benchmark.
NUM_ENV_RUNNERS = 4


sample_env = ParallelPettingZooEnv(
    FreeAgencyEnv()
)

obs_space = sample_env.observation_space["team_0"]
act_space = sample_env.action_space["team_0"]


config = (
    PPOConfig()

    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )

    .environment(
        "free_agency_v1"
    )

    .framework(
        "torch"
    )

    .resources(
        num_gpus=1
    )

    .multi_agent(
        policies={
            "shared_policy": (
                None,
                obs_space,
                act_space,
                {},
            )
        },

        policy_mapping_fn=lambda agent_id, *args, **kwargs:
            "shared_policy",
    )

    .training(
        train_batch_size=512,

        model={
            "custom_model":
                "free_agency_masked_model",
        },
    )

    .experimental(
        _disable_preprocessor_api=True
    )

    .env_runners(
        num_env_runners=NUM_ENV_RUNNERS,
        num_cpus_per_env_runner=1,
    )
)


# ============================================================
# 5. Build the algorithm
# ============================================================

print("\nBuilding PPO algorithm...")

algo = config.build_algo()

print("PPO algorithm created.")


# ============================================================
# 6. Restore EVERYTHING from the checkpoint
# ============================================================

print(f"Restoring from {CHECKPOINT_PATH}...")
algo.restore(CHECKPOINT_PATH)

print("Successfully restored checkpoint.")
print("Current iteration:", algo.iteration)



# IMPORTANT:
#
# This restores the RLlib Algorithm state rather than merely
# loading the neural-network weights.
#
# Therefore PPO resumes with the checkpoint's:
#
#   - policy weights
#   - value-function weights
#   - optimizer state
#   - PPO training state
#   - iteration counters
#   - other RLlib algorithm state
#


print("\nCheckpoint restored successfully.")


# ============================================================
# 7. Verify where we resumed
# ============================================================

print("\n" + "=" * 70)
print("RESUMED TRAINING STATE")
print("=" * 70)


print(
    f"Algorithm iteration  : "
    f"{algo.iteration}"
)

print(
    f"Target iteration     : "
    f"{TARGET_ITER}"
)

print("=" * 70)


# ============================================================
# 8. Continue training
# ============================================================

process = psutil.Process(
    os.getpid()
)

print(
    "\nMemory:",
    process.memory_info().rss / 1024**3,
    "GB",
)

print("\nTraining resumed!\n")


while algo.iteration < TARGET_ITER:

    start_time_iter = time.time()

    current_iteration = algo.iteration + 1

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"Training iteration "
        f"{current_iteration} / {TARGET_ITER}"
    )

    print(
        f"{'=' * 70}"
    )

    result = algo.train()

    elapsed = (
        time.time()
        - start_time_iter
    )

    metrics_source = result.get(
        "env_runners",
        result,
    )

    ep_reward_mean = metrics_source.get(
        "episode_reward_mean",
        np.nan,
    )

    ep_len_mean = metrics_source.get(
        "episode_len_mean",
        np.nan,
    )

    ep_count = metrics_source.get(
        "num_episodes",
        metrics_source.get(
            "episodes_this_iter",
            0,
        ),
    )

    env_steps = result.get(
        "num_env_steps_sampled",
        metrics_source.get(
            "num_env_steps_sampled",
            np.nan,
        ),
    )

    agent_steps = result.get(
        "num_agent_steps_sampled",
        metrics_source.get(
            "num_agent_steps_sampled",
            np.nan,
        ),
    )

    print(
        f"\nIteration: "
        f"{algo.iteration}"
    )

    print(
        f"Episode reward mean: "
        f"{ep_reward_mean}"
    )

    print(
        f"Episode length mean: "
        f"{ep_len_mean}"
    )

    print(
        f"Episodes completed: "
        f"{ep_count}"
    )

    print(
        f"Environment steps: "
        f"{env_steps}"
    )

    print(
        f"Agent steps: "
        f"{agent_steps}"
    )

    print(
        f"Wall time: "
        f"{elapsed:.2f} s"
    )

    if algo.iteration % 25 == 0:
        print(f"Running 10-season evaluation episode...")
        eval_metrics = evaluate_and_log_policy(algo, n_seasons=10, csv_path = "free_agency_env_win_pct.csv", iteration = algo.iteration)


    # --------------------------------------------------------
    # Periodic checkpoint
    # --------------------------------------------------------

    if algo.iteration % 100 == 0:

        print(
            "\nSaving periodic checkpoint..."
        )

        checkpoint = algo.save(
            checkpoint_dir=
                "./rllib_checkpoints/periodic"
        )

        print(
            f"Checkpoint saved to:\n"
            f"{checkpoint}"
        )


# ============================================================
# 9. Final checkpoint
# ============================================================

print(
    "\nSaving final checkpoint..."
)

final_checkpoint = algo.save(
    checkpoint_dir=
        "./rllib_checkpoints/final"
)

print(
    f"\nFinal checkpoint:\n"
    f"{final_checkpoint}"
)

print(
    "\nTraining finished."
)

