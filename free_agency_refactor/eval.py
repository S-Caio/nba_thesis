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
# Rank teams within each iteration-season pair by value
# ascending=True means rank 1 = worst (lowest win_pct); flip if you want rank 1 = best
win_pct["rank"] = (
    win_pct.groupby(["iteration", "evaluation_season"])["value"]
    .rank(method = "min", ascending = False)
)

# Re-run the worst-team marking on this ranked df (or just merge rank back into worst_team_df)
worst_team_df = mark_worst_team_season0(win_pct)

plot_df_rank = worst_team_df[
    (worst_team_df["worst_team_flag"] == True) &
    (worst_team_df["iter_group"] == worst_team_df["iter_group"].max())
]
plot_df_rank = plot_df_rank.reset_index()

#%%

# plot_df_rank
rank_plot = (
    ggplot(plot_df_rank, aes(x = "evaluation_season", y = "rank")) +
    geom_line(aes(group = "iteration"), alpha = 0.2, size = 1, color = "lightgreen") +
    geom_smooth(method = "loess", color = "forestgreen", size = 1.2, se = True) +
    scale_y_reverse() +
    labs(title = "Trajectories (in terms of rank) for the last 50 iterations")
)
rank_plot

#%%
# 1. Grab each team's rank at season 0, per iteration

win_pct_last["rank"] = (
    win_pct.groupby(["iteration", "evaluation_season"])["value"]
    .rank(method = "min", ascending = False)
)

initial_rank = (
    win_pct_last[win_pct_last["evaluation_season"] == 0]
    .loc[:, ["iteration", "variable", "rank"]]
    .rename(columns={"rank": "initial_rank"})
)

# 2. Merge that starting rank onto every row for that team/iteration
win_pct_with_initial = win_pct_last.merge(
    initial_rank, on=["iteration", "variable"], how="left"
)

# 3. Keep only the "other" seasons (exclude season 0 itself, since that's the conditioning variable)
other_seasons = win_pct_with_initial[win_pct_with_initial["evaluation_season"] != 0]

# 4. E[rank in other seasons | initial_rank]
avg_rank_by_initial = (
    other_seasons.groupby("initial_rank")["rank"]
    .agg(mean_rank="mean", std_rank="std", n="count")
    .reset_index()
)
avg_rank_by_initial["se"] = avg_rank_by_initial["std_rank"] / np.sqrt(avg_rank_by_initial["n"])

#%%
p = (
    ggplot(avg_rank_by_initial, aes(x="initial_rank", y="mean_rank")) +
    geom_ribbon(
        aes(ymin="mean_rank - se", ymax="mean_rank + se"),
        alpha=0.2
    ) +
    geom_line() +
    geom_point() +
    geom_abline(slope=1, intercept=0, linetype="dashed", color="grey") +
    xlim(0, 30) +
    ylim(0, 30) +
    labs(
        x="Rank in season 0",
        y="Average rank in subsequent seasons"
    )
)
p

#%%
# E[rank at season s | initial_rank], for each s separately
decay_df_full = (
    win_pct_with_initial.groupby(["evaluation_season", "initial_rank"])["rank"]
    .mean()
    .reset_index()
)

first_season = decay_df_full[decay_df_full["evaluation_season"] == 0]

decay_plot = (
    ggplot(decay_df_full, aes(x="evaluation_season + 1", y="rank", color="initial_rank", group="initial_rank")) +
    geom_line() +
    geom_point(data = first_season, size = 2) +
    scale_color_cmap(cmap_name="viridis") +
    scale_x_continuous(breaks = range(1, 11)) +
    labs(x="Season", y="E[rank | initial rank]")
)
decay_plot

#%%
from free_agency.utils import reward_func

win_pct["reward"] = win_pct["rank"].apply(reward_func)

episode_reward = (
    win_pct.groupby(["iteration", "variable"])["reward"]
    .sum()
    .reset_index(name="episode_reward")
)

# bring in initial_rank (rank at season 0) — reuse your earlier merge
episode_reward = episode_reward.merge(
    initial_rank, on=["iteration", "variable"], how="left"
)

learning_curve = (
    episode_reward.merge(win_pct[["iteration", "iter_group"]].drop_duplicates(), on="iteration")
    .groupby("iter_group")["episode_reward"]
    .agg(mean_reward="mean", std_reward="std", n="count")
    .reset_index()
)
learning_curve["se"] = learning_curve["std_reward"] / np.sqrt(learning_curve["n"])

p_learning_curve = (
    ggplot(learning_curve, aes(x="iter_group", y="mean_reward")) +
    geom_ribbon(aes(ymin="mean_reward - se", ymax="mean_reward + se"), alpha=0.2) +
    geom_line() +
    geom_point(size=1) +
    labs(x="Iteration group", y="Mean episode reward")
)
p_learning_curve

#%%
last_group = win_pct["iter_group"].max()

episode_reward_last = episode_reward.merge(
    win_pct[["iteration", "iter_group"]].drop_duplicates(), on="iteration"
)
episode_reward_last = episode_reward_last[episode_reward_last["iter_group"] == last_group]

reward_by_initial_rank_last = (
    episode_reward_last.groupby("initial_rank")["episode_reward"]
    .agg(mean_reward="mean", std_reward="std", n="count")
    .reset_index()
)
reward_by_initial_rank_last["se"] = reward_by_initial_rank_last["std_reward"] / np.sqrt(reward_by_initial_rank_last["n"])

p_reward_by_initial_last = (
    ggplot(reward_by_initial_rank_last, aes(x="initial_rank", y="mean_reward")) +
    geom_ribbon(aes(ymin="mean_reward - se", ymax="mean_reward + se"), alpha=0.2) +
    geom_line() +
    geom_point() +
    labs(x="Rank in season 0", y="E[total episode reward | initial rank]",
         title=f"iter_group = {last_group}")
)
p_reward_by_initial_last

#%%
win_pct_last_group = win_pct_with_initial[win_pct_with_initial["iter_group"] == last_group]

win_pct_last_group["reward"] = win_pct_last_group["rank"].apply(reward_func)

reward_decay_df_last = (
    win_pct_last_group.groupby(["evaluation_season", "initial_rank"])["reward"]
    .mean()
    .reset_index()
)

p_reward_decay_last = (
    ggplot(reward_decay_df_last, aes(x="evaluation_season + 1", y="reward", color="initial_rank", group="initial_rank")) +
    geom_line() +
    scale_color_cmap(cmap_name="viridis") +
    scale_x_continuous(breaks=range(1, 11)) +
    labs(x="Season", y="E[reward | initial rank]", title=f"iter_group = {last_group}")
)
p_reward_decay_last
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