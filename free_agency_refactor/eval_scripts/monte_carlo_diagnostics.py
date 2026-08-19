#%%
import pandas as pd
import numpy as np
from plotnine import *

#%%

df = pd.read_csv("../generated_data/episode_data.csv")


# Aggregated plot
p = (
    ggplot(df, aes(x = "win_pct")) +
    geom_density()
)

p.show()


#%%
from scipy.stats import norm
BANDWIDTH = 0.02

# Faceted by season
df["season"] = pd.Categorical(df["season"])
df["episode"] = pd.Categorical(df["episode"])

n_grid = 300
grid = np.linspace(0, 1, n_grid)


def fixed_kde(x, grid, bandwidth=BANDWIDTH):
    """
    Gaussian KDE with a fixed absolute bandwidth.
    """
    x = np.asarray(x)
    
    # Evaluate Gaussian kernel centered at every observation
    kernels = norm.pdf(
        (grid[:, None] - x[None, :]) / bandwidth
    ) / bandwidth
    
    return kernels.mean(axis=1)


episode_densities = []

for (season, episode), g in df.groupby(["season", "episode"]):
    x = g["win_pct"].dropna().to_numpy()

    if len(x) < 2 or np.ptp(x) == 0:
        continue

    episode_densities.append(
        pd.DataFrame({
            "season": season,
            "episode": episode,
            "win_pct": grid,
            "density": fixed_kde(x, grid),
            "type": "episode",
        })
    )

print("Collected episode-season densities")

episode_density_df = pd.concat(
    episode_densities,
    ignore_index=True
)

mean_density_df = (
    episode_density_df
    .groupby(
        ["season", "win_pct"],
        as_index=False,
    )
    ["density"]
    .mean()
)

overall_densities = []

for season, g in df.groupby("season"):
    x = g["win_pct"].dropna().to_numpy()

    if len(x) < 2 or np.ptp(x) == 0:
        continue

    overall_densities.append(
        pd.DataFrame({
            "season": season,
            "episode": np.nan,
            "win_pct": grid,
            "density": fixed_kde(x, grid),
            "type": "overall",
        })
    )

print("Collected overall densities")

overall_density_df = pd.concat(
    overall_densities,
    ignore_index=True
)

density_df = pd.concat(
    [episode_density_df, overall_density_df],
    ignore_index=True
)

#%%

(
    ggplot()
    + geom_line(
        data=episode_density_df,
        mapping=aes(
            x="win_pct",
            y="density",
            group="episode",
        ),
        color="#6BAFD67B",
        alpha=0.001,
        size=0.5,
    )
    + geom_line(
        data=overall_density_df,
        mapping=aes(
            x="win_pct",
            y="density",
            group="season",
        ),
        color="#08306B",
        alpha=0.9,
        size=3.0,
    )
    + facet_wrap("~season", nrow = 2)
    + labs(
        x="Win percentage",
        y="Density",
    )
    + theme_minimal()
    + theme(figure_size = (16, 10))
)

#%%
# p_facet_season = (
#     ggplot(df, aes(x="win_pct"))
#     + geom_density(
#         aes(group="factor(episode)"),
#         color="#4682B466",
#         alpha=0.05,
#         size=0.6
#     )
#     + geom_density(
#         color="navy",
#         size=1.5
#     )
#     + facet_wrap("~factor(season)")
# )
# 
# 
# (

# (
#     ggplot(df, aes(x = "win_pct"))
#     + geom_density(
#         aes(group = "episode"),
#         color = "#6EA4D361",
#         alpha=0.05,
#         size = 0.6,
#         n = 100
#         )
#     + geom_density(color = "navy", size = 1.5, alpha = 0.3)
#     + facet_wrap("~season")
# )

# p_facet_season.show()

# p_all_seasons = (
#     ggplot(df, aes(x = "win_pct", fill = "factor(season)", color = "factor(season)"))
#     + geom_density(alpha = 0.2, size = 1)
# )

# p_all_seasons.show()


#%%
# Wasserstein distance per episode-season combo density plot
from scipy.stats import wasserstein_distance, energy_distance, gaussian_kde
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon

BIN_EDGES = np.linspace(0, 1, 10)
def to_density(sample, bin_edges=BIN_EDGES, eps=1e-10):
    counts, _ = np.histogram(sample, bins=bin_edges)
    probs = counts / counts.sum()
    probs = probs + eps          # avoid exact zeros (helps KL; harmless for JSD)
    probs = probs / probs.sum()  # renormalize after adding eps
    return probs

# def to_density(sample):
#     density = transformed_kde(sample)
#     return density / density.sum()

real_data = pd.read_csv("../../initial_code/nba_team_historical_percentiles.csv")
current_data = real_data[real_data["system"] == "Current"]
old_data = real_data[real_data["system"] == "Old"]


#%%

def calc_kl(data_a, data_b):
    pa = to_density(data_a)
    pb = to_density(data_b)
    return np.sum(rel_entr(pa, pb))

def calc_jsd(data_a, data_b):
    return jensenshannon(to_density(data_a), to_density(data_b), base = 2)


def compare_seasons(
    df_sim,
    df_real,
    value_col_sim="win_pct",
    value_col_real="WinPCT",
    metric = calc_kl
):
    real_seasons = {
        real_season: data[value_col_real].to_numpy()
        for real_season, data in df_real.groupby("Season")
    }

    results = []

    for (episode, season), data in df_sim.groupby(["episode", "season"]):
        sim_values = data[value_col_sim].to_numpy()

        for real_season, real_values in real_seasons.items():
            d = metric(sim_values, real_values)

            results.append({
                "episode": str(episode),
                "season": str(season),
                "real_season": real_season,
                "metric": d,
            })

    return pd.DataFrame(results)


season_distances = compare_seasons(
    df,
    current_data,
    metric = calc_kl
)
season_distances
#%%

p_wass_by_season = (
    ggplot(season_distances, aes(x = "metric", fill = "season", color = "season")) +
    geom_density(alpha = 0.2, size = 1.5)
    + labs(x = "KL")
)

p_wass_by_season.show()

p_wass_real_seasons = (
    ggplot(season_distances, aes(x = "metric"))
    + geom_density(size = 1.5)
    + facet_grid(rows = "season", cols = "real_season")
)

p_wass_real_seasons.show()

#%% Maybe make a plot of medians here, instead of using the whole distribution
def bootstrap_median_double(
    df_sim, df_real, sim_season,
    n_boot=500, bin_edges=BIN_EDGES,
    value_col_real="WinPCT", season_col="Season",
    rng=None, metric=calc_kl
):
    rng = rng or np.random.default_rng()

    sim_subset = df_sim[df_sim["season"] == sim_season]

    episode_densities = [
        to_density(g["win_pct"].to_numpy(), bin_edges)
        for _, g in sim_subset.groupby("episode")
    ]
    n_episodes = len(episode_densities)

    season_groups = {
        s: g[value_col_real].to_numpy()
        for s, g in df_real.groupby(season_col)
    }
    season_labels = list(season_groups.keys())
    n_seasons = len(season_labels)

    # --------------------------------------------------
    # Observed median
    # --------------------------------------------------
    real_pool = np.concatenate(list(season_groups.values()))
    real_density = to_density(real_pool, bin_edges)

    observed_metric_vals = np.array([
        metric(ed, real_density)
        for ed in episode_densities
    ])

    observed_median = np.median(observed_metric_vals)

    # --------------------------------------------------
    # Bootstrap
    # --------------------------------------------------
    boot_medians = np.empty(n_boot)

    for b in range(n_boot):
        # Source 1: resample real seasons (block-intact)
        chosen = rng.choice(
            season_labels,
            size=n_seasons,
            replace=True
        )

        real_pool = np.concatenate([
            season_groups[s] for s in chosen
        ])

        real_density = to_density(real_pool, bin_edges)

        # Metric for every episode against this bootstrap
        # real distribution
        metric_vals = np.array([
            metric(ed, real_density)
            for ed in episode_densities
        ])

        # Source 2: resample episodes
        episode_idx = rng.integers(
            0, n_episodes, size=n_episodes
        )

        boot_medians[b] = np.median(
            metric_vals[episode_idx]
        )

    return observed_median, boot_medians

def bootstrap_real_only(
    df_sim, df_real, sim_season,
    n_boot=200,
    bin_edges=BIN_EDGES,
    value_col_real="WinPCT",
    season_col="Season",
    rng=None,
    metric=calc_kl
):
    rng = rng or np.random.default_rng()

    sim_subset = df_sim[df_sim["season"] == sim_season]

    episode_densities = [
        to_density(g["win_pct"].to_numpy(), bin_edges)
        for _, g in sim_subset.groupby("episode")
    ]

    season_groups = {
        s: g[value_col_real].to_numpy()
        for s, g in df_real.groupby(season_col)
    }

    season_labels = list(season_groups.keys())
    n_seasons = len(season_labels)

    results = []

    for b in range(n_boot):
        chosen = rng.choice(
            season_labels,
            size=n_seasons,
            replace=True
        )

        real_pool = np.concatenate([
            season_groups[s] for s in chosen
        ])

        real_density = to_density(real_pool, bin_edges)

        vals = np.array([
            metric(ed, real_density)
            for ed in episode_densities
        ])

        results.append(np.median(vals))

    return np.array(results)

rng = np.random.default_rng(0)

boot_results = []
observed_results = []

for season in range(10):
    print(f"Bootstrapping for season {season}")

    # observed, medians = bootstrap_median_double(
    #     df,
    #     current_data,
    #     season,
    #     n_boot=200,
    #     rng=rng,
    #     metric = calc_kl
    # )

    medians = bootstrap_real_only(df, current_data, season, rng = rng, metric = wasserstein_distance)

    

    # observed_results.append({
    #     "season": str(season),
    #     "observed_median": observed
    # })

    boot_results.append(
        pd.DataFrame({
            "season": str(season),
            "boot_median": medians
        })
    )

boot_df = pd.concat(boot_results, ignore_index=True)
observed_df = pd.DataFrame(observed_results)

p_boot = (
    ggplot(boot_df, aes(x="boot_median", fill="season", color="season"))
    + geom_density(alpha=0.2, size=1.2)
    + labs(x="Bootstrap median KL (vs pooled real seasons)", y="Density",
           fill="Season", color="Season")
    + theme_bw()
)
p_boot.show()


#%%
def bootstrap_median_disaggregated(
    df_sim,
    df_real,
    sim_season,
    n_boot=500,
    bin_edges=BIN_EDGES,
    value_col_real="WinPCT",
    season_col="Season",
    rng=None,
    metric=calc_kl,
):
    rng = rng or np.random.default_rng()

    # -----------------------------
    # Simulation episodes
    # -----------------------------
    sim_subset = df_sim[df_sim["season"] == sim_season]

    episode_densities = [
        to_density(g["win_pct"].to_numpy(), bin_edges)
        for _, g in sim_subset.groupby("episode")
    ]

    # -----------------------------
    # Real seasons
    # -----------------------------
    season_groups = {
        s: to_density(g[value_col_real].to_numpy(), bin_edges)
        for s, g in df_real.groupby(season_col)
    }

    season_labels = list(season_groups.keys())
    n_seasons = len(season_labels)

    # -----------------------------
    # Observed statistic
    # -----------------------------
    observed_vals = []

    for real_density in season_groups.values():
        for episode_density in episode_densities:
            observed_vals.append(
                metric(episode_density, real_density)
            )

    observed_median = np.median(observed_vals)

    # -----------------------------
    # Bootstrap
    # -----------------------------
    boot_medians = np.empty(n_boot)

    for b in range(n_boot):

        # Resample real seasons as blocks
        chosen = rng.choice(
            season_labels,
            size=n_seasons,
            replace=True,
        )

        # Calculate ALL episode x selected-real-season distances
        boot_vals = []

        for s in chosen:
            real_density = season_groups[s]

            for episode_density in episode_densities:
                boot_vals.append(
                    metric(episode_density, real_density)
                )

        # Median over both dimensions
        boot_medians[b] = np.median(boot_vals)

    return observed_median, boot_medians

rng = np.random.default_rng(0)

boot_results = []
observed_results = []

for season in range(10):
    print(f"Bootstrapping season {season}")

    observed, medians = bootstrap_median_disaggregated(
        df,
        current_data,
        season,
        n_boot=150,
        rng=rng,
        metric=calc_kl,
    )

    observed_results.append({
        "season": str(season),
        "observed_median": observed,
    })

    boot_results.append(
        pd.DataFrame({
            "season": str(season),
            "boot_median": medians,
        })
    )

boot_df_disagg = pd.concat(
    boot_results,
    ignore_index=True,
)

observed_df_disagg = pd.DataFrame(observed_results)

p_boot_disagg = (
    ggplot(
        boot_df_disagg,
        aes(
            x="boot_median",
            fill="season",
            color="season",
        ),
    )
    + geom_density(alpha=0.2, size=1.2)
    + labs(
        x="Bootstrap median KL",
        y="Density",
        title="Disaggregated real-season bootstrap",
        fill="Simulation season",
        color="Simulation season",
    )
    + theme_bw()
)

p_boot_disagg.show()

#%%
ci_df = (
    boot_df
    .groupby("season")["boot_median"]
    .agg(
        lower=lambda x: x.quantile(0.025),
        upper=lambda x: x.quantile(0.975),
    )
    .reset_index()
)

plot_df = observed_df.merge(ci_df, on="season")

p_boot_bar = (
    ggplot(plot_df, aes(x="season", y="observed_median"))
    + geom_col()
    + geom_errorbar(
        aes(
            ymin="lower",
            ymax="upper"
        ),
        width=0.2,
        size=1
    )
    + labs(
        x="Simulation season",
        y="Median KL",
        title="Observed median KL with 95% bootstrap interval"
    )
    + theme_bw()
)

p_boot_bar.show()

#%%
sim_df = df.copy()
sim_df["source"] = "Simulation"

c_data = current_data.copy()
c_data = c_data[["Team", "Season", "WinPCT"]]
c_data["episode"] = pd.NA
c_data = c_data.rename({"Team" : "team", "Season" : "real_season", "WinPCT" : "win_pct"}, axis = 1)
c_data["source"] = "Real"

sim_seasons = sorted(df["season"].unique())
real_seasons = sorted(c_data["real_season"].unique())

sim_plot = sim_df.merge(
    pd.DataFrame({"real_season": real_seasons}),
    how="cross"
)

real_plot = c_data.merge(
    pd.DataFrame({"season": sim_seasons}),
    how="cross"
)


plot_df = pd.concat(
    [
        sim_plot[
            ["season", "real_season", "win_pct", "source"]
        ],
        real_plot[
            ["season", "real_season", "win_pct", "source"]
        ],
    ],
    ignore_index=True
)


p = (
    ggplot(
        plot_df,
        aes(
            x="win_pct",
            color="source"
        )
    )
    + geom_density(size=1.2)
    + facet_grid(
        "season ~ real_season"
    )
    + labs(
        x="Win %",
        y="Density",
        color="Data"
    )
    + theme_bw()
)

p.show()

#%%
from scipy.stats import energy_distance, ks_2samp
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
import diptest


BIN_EDGES = np.linspace(0, 1, 10)
def to_density(sample, bin_edges=BIN_EDGES, eps=1e-10):
    counts, _ = np.histogram(sample, bins=bin_edges)
    probs = counts / counts.sum()
    probs = probs + eps          # avoid exact zeros (helps KL; harmless for JSD)
    probs = probs / probs.sum()  # renormalize after adding eps
    return probs

def calc_z_score(arr):
    return (arr - arr.mean())


def calc_distances(sim_season):
    a = current_data[current_data["Season"] == "2025-26"]["WinPCT"]
    c = current_data[current_data["Season"] == "2023-24"]["WinPCT"]
    b = df[df["season"] == sim_season]["win_pct"]

    # dip_a, pval_a = diptest.diptest(a)
    # dip_b, pval_b = diptest.diptest(b)
    # print(f"2025-26: dip={dip_a:.4f}, p={pval_a:.4f} ({'bimodal' if pval_a < 0.05 else 'not sig.'})")
    # print(f"Season {sim_season}: dip={dip_b:.4f}, p={pval_b:.4f} ({'bimodal' if pval_b < 0.05 else 'not sig.'})")


    a_adjusted = calc_z_score(a)
    b_adjusted = calc_z_score(b)

    wd = wasserstein_distance(a_adjusted, b_adjusted)
    wd2 = wasserstein_distance(a, b)
    wd3 = wasserstein_distance(c, b)
    ed = energy_distance(a, b)

    ks = ks_2samp(a, b)
    ks_c = ks_2samp(b, c)

    pa = to_density(a)
    pb = to_density(b)

    # jensenshannon returns the JS *distance* (sqrt of the divergence) by default
    js_dist = calc_jsd(a, b)
    js_div = js_dist ** 2
    js_c = calc_jsd(b, c)

    kl = calc_kl(b, a)
    kl_c = calc_kl(b, c)

    print(f"Season {sim_season}: Wasserstein={wd:.4f}, {wd2:.4f}  Energy={ed:.4f}  "
          f"JS-divergence={js_div:.4f}  JS-distance={js_dist:.4f}   KL={kl}    KS = {ks.statistic:.4f}"
          )

    print(f"Season {sim_season} (OR): Wasserstein={wd3:.4f}, JSD={js_c:.4f}, KL={kl_c:.4f}, KS={ks_c.statistic:.4f}")

for season in range(10):
    calc_distances(season)

# plot_df[plot_df["season"] == 0]