#%%
import numpy as np
import pandas as pd
import os
import torch
import ray
import time
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns



from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from free_agency.env_parallel import FreeAgencyEnv

#%%
CHECKPOINT_DIR = os.path.abspath("./rllib_checkpoints/saves")
RL_MODULE_DIR = os.path.join(CHECKPOINT_DIR, "learner_group", "learner", "rl_module", "shared_policy")

NUM_EPISODES = 1000         
NUM_WORKERS = 4              
ENVS_PER_WORKER = 16        
GPUS_PER_WORKER = 0.25       



#%%
@ray.remote
class BatchedEpisodeSampler:
    """
    Runs `num_envs` copies of the environment IN LOCKSTEP. At every step, it
    stacks that step's observations across ALL envs and ALL agents into a
    single batch and does ONE forward_inference call. This is what makes GPU
    use actually pay off: batch size = num_envs * num_agents_per_env instead
    of just num_agents_per_env.
    """

    def __init__(self, rl_module_dir, num_envs, use_gpu=True):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.rlm = RLModule.from_checkpoint(rl_module_dir).to(self.device)
        self.rlm.eval()
        self.action_dist_cls = self.rlm.get_inference_action_dist_cls()
        self.num_envs = num_envs
        self.envs = [FreeAgencyEnv() for _ in range(num_envs)]

    def _batched_actions(self, obs_list, agent_lists):
        """
        obs_list[i]    = obs dict for env i (team -> obs dict)
        agent_lists[i] = list of active agents for env i
        Returns: list of {team: action} dicts, one per env, in the same order.
        """
        flat_index = []  # (env position, team) for every row in the batch
        keys = None
        arrays = None

        for env_pos, (obs, agents) in enumerate(zip(obs_list, agent_lists)):
            for team in agents:
                if keys is None:
                    keys = obs[team].keys()
                    arrays = {k: [] for k in keys}
                flat_index.append((env_pos, team))
                for k in keys:
                    arrays[k].append(obs[team][k])

        batch = {
            k: torch.from_numpy(np.stack(v)).to(self.device)
            for k, v in arrays.items()
        }

        with torch.no_grad():
            output = self.rlm.forward_inference({Columns.OBS: batch})
        action_dist = self.action_dist_cls.from_logits(output[Columns.ACTION_DIST_INPUTS])
        actions = action_dist.sample().cpu()

        actions_per_env = [dict() for _ in range(len(obs_list))]
        for row, (env_pos, team) in enumerate(flat_index):
            actions_per_env[env_pos][team] = actions[row].item()
        return actions_per_env

    def run_episode_batch(self, episode_ids):
        """
        Runs len(episode_ids) episodes, one per env slot, to completion.
        episode_ids: list of ints (len <= self.num_envs) used to tag records.
        """
        n = len(episode_ids)
        envs = self.envs[:n]

        obs_list = []
        for env in envs:
            obs, info = env.reset()
            obs_list.append(obs)

        records = [[] for _ in range(n)]
        active = list(range(n))  # positions still running

        while active:
            agent_lists = [envs[i].agents for i in active]
            obs_subset = [obs_list[i] for i in active]
            actions_list = self._batched_actions(obs_subset, agent_lists)

            still_active = []
            for pos, i in enumerate(active):
                env = envs[i]

                if env.num_moves == 0 and env.season != 0:
                    for team in env.agents:
                        records[i].append({
                            "episode": episode_ids[i],
                            "team": team,
                            "season": env.season - 1,
                            "win_pct": obs_list[i][team]["win_pct"][0],
                        })

                new_obs, rewards, terminations, truncations, infos = env.step(actions_list[pos])
                obs_list[i] = new_obs

                if env.agents:
                    still_active.append(i)
                else:
                    for team in env.possible_agents:
                        records[i].append({
                            "episode": episode_ids[i],
                            "team": team,
                            "season": env.season - 1,
                            "win_pct": new_obs[team]["win_pct"][0],
                        })
            active = still_active

        return [r for env_records in records for r in env_records]


#%%
if __name__ == "__main__":
    start = time.time()
    ray.init()
    start_no_init = time.time()

    worker_cls = BatchedEpisodeSampler.options(num_gpus=GPUS_PER_WORKER)
    workers = [
        worker_cls.remote(RL_MODULE_DIR, num_envs=ENVS_PER_WORKER, use_gpu=GPUS_PER_WORKER > 0)
        for _ in range(NUM_WORKERS)
    ]

    # Hand out episodes to workers in chunks of ENVS_PER_WORKER, round-robin.
    episode_ids = list(range(NUM_EPISODES))
    chunks = []
    w = 0
    i = 0
    while i < len(episode_ids):
        chunk = episode_ids[i:i + ENVS_PER_WORKER]
        chunks.append((w % NUM_WORKERS, chunk))
        w += 1
        i += ENVS_PER_WORKER

    print(f"{NUM_WORKERS} workers x {ENVS_PER_WORKER} envs/worker, "
          f"{len(chunks)} batches for {NUM_EPISODES} episodes")

    futures = [workers[wi].run_episode_batch.remote(chunk) for wi, chunk in chunks]
    results = ray.get(futures)  # list of lists of records

    all_records = [record for batch_records in results for record in batch_records]
    df = pd.DataFrame(all_records)

    end = time.time()
    print(f"Time with ray initialisation: {end - start}")
    print(f"Time w/o ray initialisation {end - start_no_init}")

    print(df.head())
    df.to_csv("generated_data/episode_data.csv", index=False)

    ray.shutdown()