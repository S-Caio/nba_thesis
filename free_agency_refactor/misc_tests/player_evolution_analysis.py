#%%
import os
import numpy as np
import pandas as pd
from plotnine import *
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from free_agency.env_parallel import FreeAgencyEnv
from free_agency.constants import RATING
#%%
env = FreeAgencyEnv()
observations, info = env.reset()

# print(f"Evolve type = {env.evolve_type}")

def get_ratings(env):
    return env.league.players[:, RATING].copy()

def pick_allowed_action(action_mask):
    allowed_indices = np.where(action_mask == 1)[0]
    return np.random.choice(allowed_indices, 1)[0]



# r_dict = {"season_0" : get_ratings(env), ty}

records = []
SIM_PER_TYPE = 500
TYPES = ["v3"]

print("Starting simulations!")
for evolve_type in TYPES:
    for sim in range(SIM_PER_TYPE):
        if sim % 10 == 0:
            print(f"This is simulation {sim} of type {evolve_type}")

        env = FreeAgencyEnv()
        observations, info = env.reset()

        last_season = -1

        while env.agents:
            # Record once at the beginning of each season
            if env.season != last_season:
                ratings = get_ratings(env)

                for player, rating in enumerate(ratings):
                    records.append({
                        "sim": sim,
                        "type": evolve_type,
                        "season": env.season,
                        "player": player,
                        "rating": rating,
                    })

                last_season = env.season

            actions = {
                agent: pick_allowed_action(
                    observations[agent]["action_mask"]
                )
                for agent in env.agents
            }

            observations, rewards, terminations, truncations, infos = env.step(actions)


#%%
df = pd.DataFrame(records)
df.to_csv("simulations_ratings.csv", index = False)
#%%

df = pd.read_csv("simulations_ratings.csv")

percentiles = (
    df.groupby(["type", "season"])["rating"]
      .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
      .unstack()
      .reset_index()
      .rename(columns={
          0.10: "p10",
          0.25: "p25",
          0.50: "p50",
          0.75: "p75",
          0.90: "p90",
      })
)

percentiles_long = (
    df.groupby(["type", "season"])["rating"]
      .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
      .reset_index()
      .rename(columns={"level_2": "percentile", "rating": "rating"})
)

percentiles_long["percentile"] = percentiles_long["percentile"].map({
    0.10: "10th",
    0.25: "25th",
    0.50: "50th",
    0.75: "75th",
    0.90: "90th",
})

p_percentiles = (
    ggplot(
        percentiles_long,
        aes(
            x="season",
            y="rating",
            linetype="percentile",
        ),
    )
    + geom_line(size = 1.5)
    + facet_wrap("~type")
    + scale_linetype_manual(
        values={
            "10th": "dotted",
            "25th": "dashed",
            "50th": "solid",
            "75th": "dashed",
            "90th": "dotted",
        }
    )
    + labs(
        x="Season",
        y="Rating",
        linetype="Percentile",
    )
    + theme_minimal()
    + theme(figure_size=(14, 8))
)

p_percentiles.save("ratings_percentiles.pdf")

p_percentiles

#%%

p = (
    ggplot(percentiles, aes(x="season"))
    + geom_ribbon(
        aes(ymin="p10", ymax="p90"),
        alpha=0.20,
    )
    + geom_ribbon(
        aes(ymin="p25", ymax="p75"),
        alpha=0.35,
    )
    + geom_line(aes(y="p50"), size=1.2)
    + facet_wrap("~type")
    + labs(
        x="Season",
        y="Player rating",
    )
    + theme_minimal()
    + theme(figure_size=(14, 7))
)

p.save("rating_distribution_over_time.pdf")

#%%
percentiles["ratio_90_50"] = percentiles["p90"] / percentiles["p50"]
percentiles["ratio_90_10"] = percentiles["p90"] / percentiles["p10"]

ratio_plot = (
    ggplot(percentiles, aes(y = "ratio_90_50", x = "season", color = "type")) +
    geom_line()
)

ratio_plot.save("ratio_plot.pdf")


#%%
from free_agency.constants import LeagueConfig

env = FreeAgencyEnv()
observations, info = env.reset()

def get_ratings(env):
    return env.league.players[:, RATING].copy()

def pick_allowed_action(action_mask):
    allowed_indices = np.where(action_mask == 1)[0]
    return np.random.choice(allowed_indices, 1)[0]

def get_win_pct(env, records, noise):
    for i in env.agents:
        w = observations[i]["win_pct"]
        win_pct_dict = {
            "noise" : noise,
            "season" : env.season,
            "team" : i,
            "win_pct" : w[0]
        }
        records.append(win_pct_dict)
        
    return records

last_season = 0
records = []
NOISE_LEVELS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SIM_NUMBER = 30


for noise in NOISE_LEVELS:
    print(noise)
    for sim in range(SIM_NUMBER):
        if sim % 10 == 0:
            print(sim)

        env = FreeAgencyEnv(config = LeagueConfig(
            season_noise_scale = noise
        ))
        observations, info = env.reset()
        while env.agents:
            # Record once at the beginning of each season
                # ratings = get_ratings(env)

                # for player, rating in enumerate(ratings):
                #     records.append({
                #         "sim": sim,
                #         "type": evolve_type,
                #         "season": env.season,
                #         "player": player,
                #         "rating": rating,
                #     })

            actions = {
                agent: pick_allowed_action(
                    observations[agent]["action_mask"]
                )
                for agent in env.agents
            }

            observations, rewards, terminations, truncations, infos = env.step(actions)

            if env.season != last_season:
                last_season = env.season
                get_win_pct(env, records, noise)

#%%
df = pd.DataFrame(records)
df["noise"] = pd.Categorical(
    df["noise"].astype(str),
    categories=[str(n) for n in NOISE_LEVELS],
    ordered=True
)

# c = (df["noise"] == "1") & (df["season"] == 1) 
# df[c]
(
    ggplot(df, aes(x = "win_pct", fill = "noise"))
    + geom_density(alpha = 0.3)
    + facet_wrap("~noise")
    + xlim(0, 1)
    + theme(figure_size = (12, 8))
)