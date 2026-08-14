#%%
from pathlib import Path
import sys
import numpy as np
from time import time

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from free_agency.env_parallel import FreeAgencyEnv

#%%
env = FreeAgencyEnv(render_mode="human")
observations, infos = env.reset()

print("Teams in the beginning:")
print(env.league.teams["team_0"])

# print(observations["team_0"]["cap_projection"])
# print(infos)


def pick_allowed_action(action_mask):
    allowed_indices = np.where(action_mask == 1)[0]
    return np.random.choice(allowed_indices, 1)[0]


# for agent in env.agents:
#     print(observations[agent]["action_mask"])
season = 0
max_seasons = 9
start = time()
while env.agents:
    if env.season != season:
            print(f"This is season {env.season}")
            print(observations["team_0"]["n_players_team"])

    season = env.season

    actions = {agent: pick_allowed_action(observations[agent]["action_mask"]) for agent in env.agents}
    # print(actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)

    # print("Agents:")
    # print(env.agents)

    # print(observations["team_0"]["standing"])

end = time()
print(f"Running {max_seasons + 1} seasons with random actions took {end - start}")

obs, _ = FreeAgencyEnv().reset()

import pickle

size = len(pickle.dumps(obs["team_0"])) / 1024
print("size of an observation ", size, "KB")

import sys
obs = env.observe("team_0")
for k, v in obs.items():
    print(k, v.nbytes if hasattr(v, "nbytes") else sys.getsizeof(v))



# %%

print(len(env.league.players))

#%%
from free_agency.contracts import FREE_AGENT_MARKER, ACTION_CONTRACT_LENGTH, ACTION_SALARY, ACTION_PLAYER_ID
from free_agency.constants import TEAM, AGE
from free_agency.state import agent_to_team_id
env = FreeAgencyEnv()
observations, infos = env.reset()

def pick_allowed_action(action_mask):
    allowed_indices = np.where(action_mask == 1)[0]
    return np.random.choice(allowed_indices, 1)[0]

season = 0
for i in range(2):
    if env.season != season:
            print(f"This is season {env.season}")
            # print(observations["team_0"]["cap_projection"])

    season = env.season

    actions = {agent: pick_allowed_action(observations[agent]["action_mask"]) for agent in env.agents}
    # print(actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)


free_agent_idx = env.league.players[:, TEAM] == FREE_AGENT_MARKER
free_agents = env.league.players[free_agent_idx, :]

# print(free_agents.shape)
# print(observations["team_0"]["my_team"])
# print(observations["team_0"]["team_salary"])

print(env.league.teams["team_0"])
print(np.count_nonzero(env.league.teams["team_0"]))

def players_relative_z_score(env, eps=1e-8):
    teams = env.league.teams 
    n_players = np.array([np.count_nonzero(t) for t in teams.values()], dtype=np.float32)
    mean, std = n_players.mean(), n_players.std()

    return {
        agent: np.float32((np.count_nonzero(t) - mean) / (std + eps))
        for agent, t in teams.items()
    }

players_relative_z_score(env)

# print(np.mean(env.league.players[env.league.players[:, TEAM] == 1][:, AGE]))

# def calculate_ages(env, league):
#     ages_dict = {}
#     for agent in env.agents:
#         team_ages = league.players[league.players[:, TEAM] == agent_to_team_id(agent)][:, AGE]
#         ages_dict[agent] = np.mean(team_ages).astype(np.float32)

#     return ages_dict

# calculate_ages(env, env.league)
# %%

#%%
from free_agency.contracts import FREE_AGENT_MARKER, ACTION_CONTRACT_LENGTH, ACTION_SALARY, ACTION_PLAYER_ID
from free_agency.constants import TEAM, AGE


# %%

def pick_allowed_action(action_mask):
    allowed_indices = np.where(action_mask == 1)[0]
    return np.random.choice(allowed_indices, 1)[0]

env = FreeAgencyEnv()
obs, infos = env.reset()
for i in range(500):
    actions = {a: pick_allowed_action(obs[a]["action_mask"]) for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    for agent, o in obs.items():
        for k, v in o.items():
            if np.isnan(np.asarray(v)).any():
                print(f"NaN in {k} for {agent} at step {i}")

#%%
import seaborn as sns
import matplotlib.pyplot as plt
x = np.random.lognormal(1, 0.75, size = 1000)

print(x.min())
print(x.mean())
print(np.median(x))
print(x.max())

# print(x.percentile([0.5, 0.75, 0.9, 0.99, 0.99999]))
print(np.percentile(x, [50, 75, 90, 95, 99]))


sns.kdeplot(x)
plt.show()

# #%%
# from pathlib import Path
# import sys
# import numpy as np
# from time import time

# project_root = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(project_root))
# from free_agency.env import FreeAgencyEnv

# #%%
# env = FreeAgencyEnv(render_mode="human")
# env.reset()

# # print(observations["team_0"]["cap_projection"])
# # print(infos)



# def pick_allowed_action(action_mask):
#     allowed_indices = np.where(action_mask == 1)[0]
#     return np.random.choice(allowed_indices, 1)[0]


# env.observe("team_0")["action_mask"]

# # for agent in env.agents:
# #     print(observations[agent]["action_mask"])
# season = 0
# max_seasons = 10
# start = time()
# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         acion = None
#         print("Termination or truncation!")
#         break
#     else:
#         action = pick_allowed_action(env.observe(agent)["action_mask"])

#     env.step(action)
#     if env.season != season:
#         season = env.season
#         print(season)


# end = time()
# print(f"Running {max_seasons} seasons with random actions took {end - start}")



