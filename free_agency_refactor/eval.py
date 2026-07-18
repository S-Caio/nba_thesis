#%%
import numpy as np
import pandas as pd
from plotnine import *
# %%
win_pct = pd.read_csv("free_agency_env_win_pct.csv")
# win_pct
win_pct = win_pct.melt(
    id_vars=["iteration", "evaluation_season"]
    )

win_pct = win_pct[win_pct["variable"] != "league_std_dev"]

p = (
    ggplot(win_pct, aes(x = "value")) +
    geom_density() +
    facet_wrap("~iteration")
    # facet_grid(cols = "iteration", rows = "evaluation_season")
)
p

#%%
# Bringing in real NBA data

nba_win_pct = pd.read_csv("../initial_code/current_system_win_pct_series.csv")
nba_win_pct
(
    ggplot(nba_win_pct, aes(x = "WinPCT")) +
    geom_density()
)

#%%

# Simulated data
sim = win_pct[["value"]].copy()
sim["dataset"] = "Simulation"
sim = sim.rename(columns={"value": "win_pct"})

# NBA data
real = nba_win_pct[["WinPCT"]].copy()
real["dataset"] = "NBA"
real = real.rename(columns={"WinPCT": "win_pct"})

# Combine
plot_df = pd.concat([sim, real], ignore_index=True)

(
    ggplot(plot_df, aes(x="win_pct", color="dataset", fill="dataset"))
    + geom_density(alpha=0.3)
)

#%%
plot_df
# %%
# Unique facet combinations
facets = (
    win_pct[["iteration", "evaluation_season"]]
    .drop_duplicates()
)

# Cross join
nba_facet = (
    facets.merge(
        nba_win_pct[["WinPCT"]],
        how="cross"
    )
    .rename(columns={"WinPCT": "value"})
)
nba_facet["dataset"] = "NBA"

sim = win_pct.copy()
sim["dataset"] = "Simulation"

plot_df = pd.concat([sim, nba_facet], ignore_index=True)

p = (
    ggplot(
        plot_df,
        aes(x="value", color="dataset", fill="dataset")
    )
    + geom_density(alpha=0.3)
    + facet_wrap("~iteration")
)

# display(p)
#%%
from scipy.stats import wasserstein_distance

# Real distribution
real = nba_win_pct["WinPCT"].to_numpy()

# Compute Wasserstein distance for each iteration/season
wasserstein_df = (
    win_pct
    .groupby(["iteration"])
    .agg(
        wasserstein=(
            "value",
            lambda x: wasserstein_distance(x.to_numpy(), real)
        )
    )
    .reset_index()
)

wasserstein_df

(
    ggplot(
        wasserstein_df,
        aes(
            x="iteration",
            y="wasserstein"
            )
    )
    + geom_line()
    + geom_point()
    + labs(
        x="Iteration",
        y="Wasserstein distance"
    )
    + theme_bw()
)