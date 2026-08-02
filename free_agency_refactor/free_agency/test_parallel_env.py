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
while env.season <= max_seasons:
    if env.season != season:
            print(f"This is season {env.season}")

    season = env.season

    actions = {agent: pick_allowed_action(observations[agent]["action_mask"]) for agent in env.agents}
    print(actions)

    observations, rewards, terminations, truncations, infos = env.step(actions)

end = time()
print(f"Running {max_seasons + 1} seasons with random actions took {end - start}")

    


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


    
