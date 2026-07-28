#%%
import numpy as np
import pandas as pd
from plotnine import *
# %%
win_pct = pd.read_csv("free_agency_env_win_pct.csv")
# win_pct[win_pct["iteration"] == 1037]
win_pct = win_pct.melt(
    id_vars=["iteration", "evaluation_season"]
    )

win_pct = win_pct[win_pct["variable"] != "league_std_dev"]

win_pct["iter_group"] = win_pct["iteration"] // 100

#%%
# Alternative reading script (in case file was appended instead of overwritten)

win_pct = pd.read_csv("free_agency_env_win_pct.csv")

start_of_obs = win_pct[win_pct["iteration"] == win_pct["iteration"].max()].index.stop # Assuming old run went on for longer than current run.
win_pct = win_pct.iloc[start_of_obs:, :]
win_pct = win_pct.reset_index(drop = True)
win_pct = win_pct.melt(
    id_vars=["iteration", "evaluation_season"]
    )

win_pct = win_pct[win_pct["variable"] != "league_std_dev"]

win_pct["iter_group"] = win_pct["iteration"] // 100
win_pct
#%%
p = (
    ggplot(win_pct, aes(x = "value")) +
    geom_density() +
    facet_wrap("~iter_group")
    # facet_grid(cols = "iteration", rows = "evaluation_season")
)
p


#%%


win_pct_last = win_pct[win_pct["iter_group"] == win_pct["iter_group"].max()]

p_last_group = (
    ggplot(win_pct_last, aes(x = "value")) +
    geom_density() +
    facet_wrap("~evaluation_season")

)

p_last_group

#%%
def mark_worst_team_season0(df):
    df = df.sort_values(["iteration", "evaluation_season"]).copy()

    def mark_group(g):
        season0 = g[g["evaluation_season"] == 0]
        worst_team = season0.loc[season0["value"].idxmin(), "variable"]

        g["worst_team"] = worst_team
        g["worst_team_flag"] = g["variable"] == worst_team
        return g

    return df.groupby("iteration", group_keys=True).apply(mark_group)

worst_team_df = mark_worst_team_season0(win_pct)
plot_df = worst_team_df[(worst_team_df["worst_team_flag"] == True) & (worst_team_df["iter_group"] == worst_team_df["iter_group"].max())]
plot_df.reset_index(inplace = True)
time_plot = (
    ggplot(plot_df, aes(x = "evaluation_season", y = "value")) +
    geom_line(aes(group = "iteration"), color = "steelblue", alpha = 0.2, size = 0.6) +
    geom_smooth(method = "loess", color = "navy", size = 1.2, se = True) +
    labs(title = "Trajectory of worst team in last 50 updates")
)
time_plot
# plot_df
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