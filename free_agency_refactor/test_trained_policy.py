import os
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# Adjust these imports to match your project structure
from free_agency.env import FreeAgencyEnv
from free_agency.constants import LeagueConfig
from train import FreeAgencyMaskedModel 

# 1. Re-register your custom model architectural mapping
ModelCatalog.register_custom_model("free_agency_masked_model", FreeAgencyMaskedModel)

def env_creator(config):
    return PettingZooEnv(FreeAgencyEnv())

def decode_action(action, config):
    n_lengths = len(config.contract_lengths)
    n_salaries = len(config.salary_ranges)
    n_contract_actions = config.n_players * n_salaries * n_lengths
    
    if action == n_contract_actions:
        return "Pass (Declined to offer any contract)"
    
    rem = action
    length_idx = rem % n_lengths
    rem //= n_lengths
    salary_idx = rem % n_salaries
    player_idx = rem // n_salaries
    
    return f"Offer to Player ID {player_idx:03d} -> {config.contract_lengths[length_idx]} Yrs @ ${config.salary_ranges[salary_idx]:,.2f}/yr"


def inspect_trained_policy(checkpoint_path):
    checkpoint_path = checkpoint_path = os.path.abspath(checkpoint_path)

    if not ray.is_initialized():
        # Keep things strictly pinned to 1 CPU core for local inference evaluation
        ray.init(num_cpus=1, num_gpus=0)
        
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
    algo = config_obj.build()
    algo.restore(checkpoint_path)
    
    # 3. Spin up an evaluation environment instance
    config = LeagueConfig()
    env = FreeAgencyEnv(config=config)
    env.reset()
    
    print("\n==========================================================================")
    print(" 🎬 STARTING HUMAN-READABLE INFERENCE SIMULATION")
    print("==========================================================================\n")
    
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            env.step(None)
            continue
            
        action = algo.compute_single_action(obs, policy_id="shared_policy", explore=False)
        
        readable_offer = decode_action(action, config)
        team_display_name = env.agent_name_mapping.get(agent, agent)
        allowed_moves_count = int(np.sum(obs["action_mask"]))
        
        print(f"👉 {team_display_name:<22} | Actions Allowed: {allowed_moves_count:5d} | Action Taken: {readable_offer}")
        
        env.step(action)
        
        if env.num_moves == 0:
            print("\n==========================================================================")
            print(" 🏁 MARKET RESOLVED: CURRENT CAP UTILIZATION SUMMARY")
            print("==========================================================================\n")
            for team_id in env.possible_agents:
                real_name = env.agent_name_mapping.get(team_id, team_id)
                cap_used = env.league.team_salaries[team_id]
                cap_max = config.salary_cap
                print(f" 📊 {real_name:<22} | Salary Allocated: ${cap_used:11,.2f} / ${cap_max:11,.2f} ({(cap_used/cap_max)*100:6.2f}%)")
            print("\n=========fe=================================================================")
            break
            
    ray.shutdown()

if __name__ == "__main__":
    FINAL_CHECKPOINT = "./rllib_checkpoints/final"
    inspect_trained_policy(FINAL_CHECKPOINT)