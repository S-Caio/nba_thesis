import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
import os

from free_agent_environment import FreeAgencyEnv


def env_creator(config):
    """
    Factory for Ray. No SuperSuit needed — RLlib's FlattenObservations
    connector will handle the Dict -> Box conversion in the data pipeline.
    """
    raw_env = FreeAgencyEnv(n_teams=30, players_per_team=10, n_seasons=10)
    return PettingZooEnv(raw_env)  # AEC wrapper, no flatten_v0


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    env_name = "nba_free_agency_aec"
    register_env(env_name, env_creator)

    # Peek at the raw (un-flattened) spaces just for logging — RLlib will
    # automatically account for the connector when building the network.
    _tmp_env = env_creator({})
    print(f"Raw Observation Space: {_tmp_env.observation_space}")
    print(f"Action Space:          {_tmp_env.action_space}")

    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .env_runners(
            num_env_runners=1,
            # KEY FIX: FlattenObservations connector replaces supersuit.flatten_v0.
            # It flattens Dict obs into a single Box before they hit the network.
            env_to_module_connector=lambda env, spaces, device: FlattenObservations(
                multi_agent=True
            ),
        )
        .resources(num_gpus=0)
        .multi_agent(
            # Pass a plain set of policy names — RLlib infers the spaces
            # automatically (including the flattened shape from the connector).
            policies={"shared_policy"},
            # Newer RLlib signature; **kwargs absorbs any extra keyword args.
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .training(
            lr=3e-4,
            train_batch_size=32000,
            minibatch_size=128,
            num_sgd_iter=10,
        )
    ) 

    algo = config.build()

    print("\nStarting Ray RLlib Multi-Agent Training Loop...\n")
    for iteration in range(50):
        result = algo.train()

        print(
            f"Iteration: {iteration:02d} | "
            f"Mean Episode Reward: {result.get('env_runner_results', {}).get('episode_return_mean', 0):.4f} | "
            f"Timesteps Total: {result.get('num_env_steps_sampled_lifetime', 0)}"
        )

        if iteration % 10 == 0:
            checkpoint_dir = algo.save(
                checkpoint_dir=os.path.abspath("./rllib_checkpoints")
            )
            print(f"--> Checkpoint saved at: {checkpoint_dir}")

    final_dir = algo.save(checkpoint_dir= os.path.abspath("./rllib_models_final"))

    print(f"\nTraining Complete! Final model saved at {final_dir}")
    ray.shutdown()