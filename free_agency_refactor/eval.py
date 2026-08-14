#%%
import numpy as np
import pandas as pd
from plotnine import *
from datetime import datetime

today = datetime.today().strftime('%d_%m')
# %%
win_pct = pd.read_csv("free_agency_env_win_pct.csv")
# win_pct[win_pct["iteration"] == 1037]
win_pct = win_pct.melt(
    id_vars=["trajectory", "iteration", "evaluation_season"]
    )

win_pct = win_pct[win_pct["variable"] != "league_std_dev"]

win_pct["iter_group"] = win_pct["iteration"] // 100
win_pct

#%%
# Alternative reading script (in case file was appended instead of overwritten)

# win_pct = pd.read_csv("free_agency_env_win_pct.csv")

# start_of_obs = win_pct[win_pct["iteration"] == win_pct["iteration"].max()].index.stop # Assuming old run went on for longer than current run.
# win_pct = win_pct.iloc[start_of_obs:, :]
# win_pct = win_pct.reset_index(drop = True)
# win_pct = win_pct.melt(
#     id_vars=["iteration", "evaluation_season"]
#     )

# win_pct = win_pct[win_pct["variable"] != "league_std_dev"]

# win_pct["iter_group"] = win_pct["iteration"] // 100
# win_pct
#%%
p = (
    ggplot(win_pct, aes(x = "value")) +
    geom_density() +
    facet_wrap("~iter_group")
    # facet_grid(cols = "iteration", rows = "evaluation_season")
)
p

#%%
last_iter = win_pct[win_pct["iteration"] == win_pct["iteration"].max()]

last_iter_plot = (
    ggplot(last_iter, aes(x = "value")) +
    geom_density() +
    facet_grid(rows = "evaluation_season", cols = "trajectory")
    + theme(figure_size=(16, 12))
)

last_iter_plot

#%%

last_iter = last_iter.copy()
last_iter["trajectory"] = last_iter["trajectory"].astype(str)

last_iter_densities = (
    ggplot(last_iter, aes(x="value"))
    + geom_density(
        aes(group="trajectory"),
        color="#4682B466",
        alpha=0.05,
        size=0.6
    )
    + geom_density(
        color="navy",
        size=1.5
    )
    + facet_wrap("~evaluation_season")
)

last_iter_densities.show()

last_iter_densities.save(f"../project_diary/figs/last_iter_densities_{today}.pdf")



#%%


win_pct_last = win_pct[(win_pct["iter_group"] == win_pct["iter_group"].max())]
win_pct_last["trajectory"] = win_pct_last["trajectory"].astype(str)

p_last_group = (
    ggplot(win_pct_last, aes(x = "value")) 
    + geom_density(aes(color = "trajectory"),
                   size = 0.5)
    + geom_density(size = 2)
    + facet_wrap("~evaluation_season")

)

p_last_group

#%%
def mark_worst_team_season0(df):
    df = df.sort_values(["trajectory", "iteration", "evaluation_season"]).copy()

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

plot_df["iter_traj"] = (
    plot_df["iteration"].astype(str) + "_" +
    plot_df["trajectory"].astype(str)
)

time_plot = (
    ggplot(plot_df, aes(x = "evaluation_season", y = "value")) +
    geom_line(aes(group = "iter_traj"), color = "steelblue", alpha = 0.2, size = 0.6) +
    geom_smooth(method = "loess", color = "navy", size = 1.2, se = True) +
    labs(title = "Trajectory of worst team in last 50 updates")
)
time_plot
# plot_df

#%%
def mark_worst_k_teams_season0(df, k=4):
    df = df.sort_values(["trajectory", "iteration", "evaluation_season"]).copy()

    def mark_group(g):
        season0 = g[g["evaluation_season"] == 0]
        worst_teams = season0.nsmallest(k, "value")["variable"].tolist()

        g["worst_teams"] = [worst_teams] * len(g)
        g["worst_team_flag"] = g["variable"].isin(worst_teams)
        # rank: 1 = worst, 2 = second worst, etc. (NaN if not in top-k)
        rank_map = {team: i + 1 for i, team in enumerate(worst_teams)}
        g["worst_rank"] = g["variable"].map(rank_map)
        return g

    return df.groupby("iteration", group_keys=True).apply(mark_group)

worst_k_df = mark_worst_k_teams_season0(win_pct, k=4)
plot_df = worst_k_df[
    (worst_k_df["worst_team_flag"] == True) &
    (worst_k_df["iter_group"] == worst_k_df["iter_group"].max())
]
plot_df.reset_index(inplace=True)
plot_df["iter_traj"] = (
    plot_df["iteration"].astype(str) + "_" +
    plot_df["trajectory"].astype(str)
)
plot_df["line_id"] = plot_df["iteration"].astype(str) + "_" + plot_df["variable"].astype(str) + plot_df["trajectory"].astype(str)

time_plot = (
    ggplot(plot_df, aes(x="evaluation_season", y="value"))
    + geom_line(aes(group="line_id"), color="steelblue", alpha=0.1, size=0.5)
    + geom_smooth(aes(group="worst_rank", color="factor(worst_rank)"), method="loess", size=1.2, se=True)
    + labs(title="Trajectory of top-4 worst teams in last 50 updates", color="Worst rank", y = "Win %")
)
time_plot


#%%
# Rank teams within each iteration-season pair by value
# ascending=True means rank 1 = worst (lowest win_pct); flip if you want rank 1 = best
win_pct["rank"] = (
    win_pct.groupby(["trajectory", "iteration", "evaluation_season"])["value"]
    .rank(method = "min", ascending = False)
)

# Re-run the worst-team marking on this ranked df (or just merge rank back into worst_team_df)
worst_team_df = mark_worst_team_season0(win_pct)

plot_df_rank = worst_team_df[
    (worst_team_df["worst_team_flag"] == True) &
    (worst_team_df["iter_group"] == worst_team_df["iter_group"].max())
]
plot_df_rank = plot_df_rank.reset_index()
plot_df_rank["iter_traj"] = (
    plot_df_rank["iteration"].astype(str) + "_" +
    plot_df_rank["trajectory"].astype(str)
)

#%%

# plot_df_rank
rank_plot = (
    ggplot(plot_df_rank, aes(x = "evaluation_season", y = "rank")) +
    geom_line(aes(group = "iter_traj"), alpha = 0.1, size = 1, color = "green") +
    geom_smooth(method = "loess", color = "forestgreen", size = 1.2, se = True) +
    scale_y_reverse() +
    labs(title = "Trajectories (in terms of rank) for the last 50 iterations")
)
rank_plot

#%%
# 1. Grab each team's rank at season 0, per iteration

win_pct_last["rank"] = (
    win_pct.groupby(["trajectory", "iteration", "evaluation_season"])["value"]
    .rank(method = "min", ascending = False)
)

initial_rank = (
    win_pct_last[win_pct_last["evaluation_season"] == 0]
    .loc[:, ["trajectory", "iteration", "variable", "rank"]]
    .rename(columns={"rank": "initial_rank"})
)

# 2. Merge that starting rank onto every row for that team/iteration
win_pct_with_initial = win_pct_last.merge(
    initial_rank, on=["trajectory", "iteration", "variable"], how="left"
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
season2 = other_seasons[other_seasons["evaluation_season"] == 1]
avg_season2_rank = (
    season2.groupby("initial_rank")["rank"]
    .agg(mean_rank="mean", std_rank="std", n="count")
    .reset_index()
)
avg_season2_rank["se"] = avg_season2_rank["std_rank"] / np.sqrt(avg_season2_rank["n"])

p = (
    ggplot(avg_season2_rank, aes(x="initial_rank", y="mean_rank")) +
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
        y="Average rank in second season"
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

real_data = pd.read_csv("../initial_code/nba_team_historical_percentiles.csv")
nba_curr_df = real_data[real_data["system"] == "Current"]
nba_win_pct_curr = pd.DataFrame(nba_curr_df["WinPCT"])
p_curr = (
    ggplot(nba_win_pct_curr, aes(x = "WinPCT")) +
    geom_density()
    + xlim(0, 1)
)

nba_old_df = real_data[real_data["system"] == "Old"]
nba_win_pct_old = pd.DataFrame(nba_old_df["WinPCT"])
nba_win_pct_old

p_old = (
    ggplot(nba_win_pct_old, aes(x = "WinPCT")) +
    geom_density()
    + xlim(0, 1)
)

display(p_curr)
display(p_old)
#%%

# Simulated data
sim = win_pct_last[["value"]].copy()
sim["dataset"] = "Simulation"
sim = sim.rename(columns={"value": "win_pct"})

# NBA data
real = nba_win_pct_curr[["WinPCT"]].copy()
real["dataset"] = "NBA Current"
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
    win_pct_last[["trajectory", "iteration", "evaluation_season"]]
    .drop_duplicates()
)

# Cross join
nba_facet = (
    facets.merge(
        nba_win_pct_curr[["WinPCT"]],
        how="cross"
    )
    .rename(columns={"WinPCT": "value"})
)
nba_facet["dataset"] = "NBA"

sim = win_pct_last.copy()
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

display(p)

#%%
from scipy.stats import wasserstein_distance

# Real distribution
real_curr = nba_win_pct_curr["WinPCT"].to_numpy()

real_old = nba_win_pct_old["WinPCT"].to_numpy()

def calc_wasserstein_df(real_df, df_simmed, comparison_point=None,
                          season_col="Season", value_col="WinPCT"):
    real_seasons = {
        season: grp[value_col].to_numpy()
        for season, grp in real_df.groupby(season_col)
    }

    def per_season_distances(sim_values):
        sim_values = sim_values.to_numpy()
        return pd.Series({
            season: wasserstein_distance(sim_values, season_data)
            for season, season_data in real_seasons.items()
        })

    # long format: one row per trajectory x iteration x real season
    detail_df = (
        df_simmed
        .groupby(["trajectory", "iteration"])["value"]
        .apply(per_season_distances)
        .rename_axis(index=["trajectory", "iteration", "real_season"])
        .reset_index(name="wasserstein")
    )

    if comparison_point:
        detail_df["comparison_point"] = comparison_point

    # aggregated: mean across real seasons, per trajectory x iteration
    agg_df = (
        detail_df
        .groupby(["trajectory", "iteration"] + (["comparison_point"] if comparison_point else []))
        .agg(wasserstein=("wasserstein", "mean"))
        .reset_index()
    )

    return detail_df, agg_df

detail_curr, wasserstein_df_curr = calc_wasserstein_df(nba_curr_df, win_pct, comparison_point="Current")
detail_old, wasserstein_df_old = calc_wasserstein_df(nba_old_df,  win_pct, comparison_point="Old")

detail_df_all = pd.concat([detail_curr, detail_old])
wasserstein_df_all = pd.concat([wasserstein_df_curr, wasserstein_df_old])


summary = (
    wasserstein_df_all
    .groupby(["iteration", "comparison_point"])
    .agg(
        wasserstein_mean=("wasserstein", "mean"),
        wasserstein_sd=("wasserstein", "std"),
        n=("wasserstein", "count")
    )
    .assign(
        se=lambda d: d.wasserstein_sd / np.sqrt(d.n),
        lower=lambda d: d.wasserstein_mean - 1.96 * d.se,
        upper=lambda d: d.wasserstein_mean + 1.96 * d.se,
    )
    .reset_index()
)

summary

p_wass = (
    ggplot(summary, aes("iteration", "wasserstein_mean", color = "comparison_point"))
    + geom_ribbon(
        aes(ymin="lower", ymax="upper", fill = "comparison_point"),
        alpha=0.2
    )
    + geom_line(size=1)
    + geom_point()
    + labs(
        x="Training iteration",
        y="Wasserstein distance"
    )
    + theme_bw(base_size = 18)
    + theme(figure_size = (12, 8))
)

display(p_wass)
p_wass.save(f"../project_diary/figs/wassertein_two_comparison_{today}.pdf")


(
    detail_df_all
    .groupby(["real_season", "comparison_point"])
    .agg(mean_wasserstein=("wasserstein", "mean"))
    .reset_index()
)

# detail_df_all["real_season"] = pd.
#%%
def order_seasons(seasons):
    """Given an iterable of season strings like '2019-20', return them
    sorted chronologically by start year."""
    unique_seasons = pd.Series(seasons).unique()
    start_year = pd.Series(unique_seasons).str.slice(0, 4).astype(int)
    return list(unique_seasons[start_year.argsort()])

season_order = order_seasons(detail_df_all["real_season"])

detail_df_all["real_season"] = pd.Categorical(
    detail_df_all["real_season"],
    categories=season_order,
    ordered=True
)


detail_df_all["iter_group"] = detail_df_all["iteration"] // 100
detail_df_last = detail_df_all[detail_df_all["iter_group"] == detail_df_all["iter_group"].max()]


season_summary = (
    detail_df_last
    .groupby(["real_season", "comparison_point"], observed=True)
    .agg(
        wasserstein_mean=("wasserstein", "mean"),
        wasserstein_sd=("wasserstein", "std"),
        n=("wasserstein", "count")
    )
    .assign(
        se=lambda d: d.wasserstein_sd / np.sqrt(d.n),
        lower=lambda d: d.wasserstein_mean - 1.96 * d.se,
        upper=lambda d: d.wasserstein_mean + 1.96 * d.se,
    )
    .reset_index()
)

p_wass_season = (
    ggplot(season_summary, aes("real_season", "wasserstein_mean", color="comparison_point", group="comparison_point"))
    + geom_ribbon(
        aes(ymin="lower", ymax="upper", fill="comparison_point"),
        alpha=0.2
    )
    + geom_line(size=1)
    + geom_point()
    + labs(
        x="Real season",
        y="Wasserstein distance"
    )
    + theme_bw(base_size=18)
    + theme(figure_size=(12, 8), axis_text_x=element_text(rotation=45, hjust=1))
)

display(p_wass_season)
p_wass_season.save(f"../project_diary/figs/wasserstein_time_plot_{today}.pdf")

#%%
sim_df = last_iter.copy()

sim_df["evaluation_season"] = sim_df["evaluation_season"].astype(int)
sim_df["source"] = "Simulated"


# ------------------------------------------------------------
# Real NBA data
# ------------------------------------------------------------

real_df = nba_curr_df[["Season", "WinPCT"]].copy()

real_df = real_df.rename(
    columns={"WinPCT": "value"}
)

real_df["source"] = "Real"
real_df["real_Season"] = real_df["Season"]

real_df = real_df.drop(columns="Season")


# ------------------------------------------------------------
# Get all seasons
# ------------------------------------------------------------

sim_seasons = sorted(sim_df["evaluation_season"].unique())
real_seasons = sorted(real_df["real_Season"].unique())


# ------------------------------------------------------------
# Replicate simulated distributions across every
# real NBA season
# ------------------------------------------------------------

sim_plot = sim_df.merge(
    pd.DataFrame({"real_Season": real_seasons}),
    how="cross"
)


# ------------------------------------------------------------
# Replicate real distributions across every
# simulated evaluation season
# ------------------------------------------------------------

real_plot = real_df.merge(
    pd.DataFrame({"evaluation_season": sim_seasons}),
    how="cross"
)


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

plot_df = pd.concat(
    [
        sim_plot[
            ["evaluation_season", "real_Season", "value", "source"]
        ],
        real_plot[
            ["evaluation_season", "real_Season", "value", "source"]
        ],
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

p = (
    ggplot(
        plot_df,
        aes(
            x="value",
            color="source"
        )
    )
    + geom_density(size=1.2)
    + facet_grid(
        "evaluation_season ~ real_Season"
    )
    + labs(
        x="WinPCT",
        y="Density",
        color="Data"
    )
    + theme_bw()
)

p.show()

# p = (
#     ggplot(
#         plot_df,
#         aes(
#             x="value",
#             color="source"
#         )
#     )
#     + geom_density(size=1.2)
#     + facet_grid(
#         "evaluation_season ~ real_Season"
#     )
#     + labs(
#         x="WinPCT",
#         y="Density",
#         color="Data"
#     )
#     + theme_bw()
# )

# p.show()

#%%
wasserstein_df = (
    win_pct
    .groupby(["iteration", "evaluation_season", "trajectory"])
    .agg(
        wasserstein=(
            "value",
            lambda x: wasserstein_distance(x.to_numpy(), real_curr)
        )
    )
    .reset_index()
)

summary = (
    wasserstein_df
    .groupby(["iteration", "evaluation_season"])
    .agg(
        wasserstein_mean=("wasserstein", "mean"),
        wasserstein_sd=("wasserstein", "std"),
        n=("wasserstein", "count")
    )
    .assign(
        se=lambda d: d["wasserstein_sd"] / np.sqrt(d["n"]),
        lower=lambda d: d["wasserstein_mean"] - 1.96 * d["se"],
        upper=lambda d: d["wasserstein_mean"] + 1.96 * d["se"],
    )
    .reset_index()
)

(
    ggplot(
        summary,
        aes(
            x="iteration",
            y="wasserstein_mean",
        )
    )
    + geom_ribbon(
        aes(ymin="lower", ymax="upper"),
        alpha=0.2
    )
    + geom_line(size=0.8)
    + geom_point(size=1.2)
    + facet_wrap("~evaluation_season", ncol=5)
    + labs(
        x="Training iteration",
        y="Wasserstein distance",
    )
    + theme_bw()
    + theme(
        figure_size=(14, 6),
        subplots_adjust={"wspace": 0.25, "hspace": 0.3},
    )
)
