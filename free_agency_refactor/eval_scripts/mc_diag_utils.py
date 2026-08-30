#%%
import pandas as pd
import numpy as np
from plotnine import *

#%%


def clean_real_data(df):
    df = df[["Team", "Season", "WinPCT"]].copy()
    df["episode"] = pd.NA
    df = df.rename({"Team" : "team", "Season" : "real_season", "WinPCT" : "win_pct"}, axis = 1)
    df["source"] = "Real"
    return df

def select_seasons(df, seasons: list, type = "real"):
    if type == "real":
        season_name = "real_season"
        # win_pct_name = "win_pct"
    elif type == "sim":
        season_name = "season"
        # win_pct_name = "win_pct"

    return df[df[season_name].isin(seasons)].copy()




######################################################################################################
# Comparison type: Many simulated seasons vs one realisation of a real NBA season                    #
#                                                                                                    #
######################################################################################################
def make_long_frame(sim_df, real_df, real_seasons : list[str], sim_seasons : list[str]):
    sim_df = select_seasons(sim_df, seasons = sim_seasons, type = "sim")
    real_df = select_seasons(real_df, seasons = real_seasons, type = "real")

    sim_plot = sim_df.merge(
        pd.DataFrame({"real_season" : real_seasons}),
        how = "cross"
    )
    sim_plot["source"] = "Sim"


    real_plot = real_df.merge(
        pd.DataFrame({"season" : sim_seasons}),
        how = "cross"
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

    return plot_df


def plot_aggregate_kdes_vs_seasons(sim_df, real_df, real_seasons, sim_seasons):
    plot_df = make_long_frame(sim_df, real_df, real_seasons = real_seasons, sim_seasons = sim_seasons)

    p = (
        ggplot(
            plot_df,
            aes(
                x="win_pct",
                color="source"
            )
        )
        + geom_density(size=2)
        + facet_grid(
            "season ~ real_season"
        )
        + labs(
            x="Win %",
            y="Density",
            color="Data"
        )
        + theme_bw(base_size = 18)
        + theme(figure_size=(18, 13), legend_position="bottom")
    )

    return p


#%%
from scipy.stats import wasserstein_distance, energy_distance, ks_2samp
from diptest import diptest

def calc_distance(series1, series2, distance_func):
    dist = distance_func(series1, series2)
    return dist if isinstance(dist, np.floating) else dist.statistic

def scaled_dipstat(series):
    """Dip statistic and calibrated p-value; dip scaled by sqrt(n) for cross-n comparability."""
    dip, pval = diptest(series)
    n = len(series)
    return dip * np.sqrt(n), pval, n

def shape_distance(series1, series2, distance_func):
    """Distance after standardizing (location/scale removed, isolates shape)."""
    z1 = (series1 - series1.mean()) / series1.std()
    z2 = (series2 - series2.mean()) / series2.std()
    dist = distance_func(z1, z2)
    return dist if isinstance(dist, np.floating) else dist.statistic

def shape_centred(series1, series2, distance_func):
    """Distance after standardizing (location/scale removed, isolates shape)."""
    z1 = (series1 - series1.mean())
    z2 = (series2 - series2.mean())
    dist = distance_func(z1, z2)
    return dist if isinstance(dist, np.floating) else dist.statistic

def distance_between_aggregate_kdes(sim_df, real_df, real_seasons, sim_seasons, funcs):
    results = []

    for r_s in real_seasons:
        season_series_real = select_seasons(real_df, seasons=[r_s], type="real")["win_pct"]
        dip_real, pval_real, n_real = scaled_dipstat(season_series_real)

        for s_s in sim_seasons:
            season_series_sim = select_seasons(sim_df, seasons=[s_s], type="sim")["win_pct"]
            dip_sim, pval_sim, n_sim = scaled_dipstat(season_series_sim)

            print("##################################################")
            print(f"Real {r_s} (n={n_real}): scaled dip={dip_real:.4f}, p={pval_real:.4f}")
            print(f"Sim {s_s} (n={n_sim}): scaled dip={dip_sim:.4f}, p={pval_sim:.4f}")
            print(f"Scaled dip |diff| = {abs(dip_real - dip_sim):.4f}, p-value |diff| = {abs(pval_real - pval_sim):.4f}")

            row = {
                "real_season": r_s,
                "sim_season": s_s,
                "n_real": n_real,
                "n_sim": n_sim,
                "dip_scaled_diff": abs(dip_real - dip_sim),
                "pval_diff": abs(pval_real - pval_sim),
                "pval_real": pval_real,
                "pval_sim": pval_sim,
            }

            for func_name, func in funcs.items():
                dist = calc_distance(season_series_real, season_series_sim, distance_func=func)
                row[f"{func_name}_raw"] = dist
                row[f"{func_name}_shape"] = shape_distance(season_series_real, season_series_sim, func)
                row[f"{func_name}_centred"] = shape_centred(season_series_real, season_series_sim, func)

            results.append(row)

    results_df = pd.DataFrame(results)
    return results_df




#%%
# real_seasons = ["2012-13" ,"2017-18", "2025-26"]



def extract_stat(dist):
    """Normalize scipy return types (some return a float, KS returns a result object) to a plain float."""
    return float(dist) if isinstance(dist, (int, float, np.floating)) else float(dist.statistic)

def calc_distance(series1, series2, distance_func):
    return extract_stat(distance_func(series1, series2))

def shape_distance(series1, series2, distance_func):
    """Distance after standardizing both series (removes location/scale, isolates shape)."""
    z1 = (series1 - series1.mean()) / series1.std()
    z2 = (series2 - series2.mean()) / series2.std()
    return extract_stat(distance_func(z1, z2))

def shape_centred(series1, series2, distance_func):
    """Distance after standardizing (location/scale removed, isolates shape)."""
    z1 = (series1 - series1.mean())
    z2 = (series2 - series2.mean())
    dist = distance_func(z1, z2)
    return dist if isinstance(dist, np.floating) else dist.statistic

def shape_mod_distance(series1, series2, distance_func):
    """Distance after standardizing (location/scale removed, isolates shape)."""
    z1 = (series1 - series1.mean()) / (series1.std() ** 0.75)
    z2 = (series2 - series2.mean()) / (series2.std() ** 0.75)
    dist = distance_func(z1, z2)
    return dist if isinstance(dist, np.floating) else dist.statistic


def distribution_of_distance(sim_df, real_df, real_seasons: list, sim_seasons: list, funcs):
    """
    Returns a long dataframe of distances between each real season and every
    simulated episode/season, for every metric, split into 'raw' and 'shape'
    (standardized) distance types.

    Columns: real_season, sim_season, episode, metric, distance_type, distance
    """
    sim_df = select_seasons(sim_df, seasons=sim_seasons, type="sim")
    records = []

    for r_s in real_seasons:
        real_season_series = select_seasons(real_df, seasons=[r_s], type="real")["win_pct"]

        for (episode, s_s), group in sim_df.groupby(["episode", "season"]):
            sim_series = group["win_pct"]

            for func_name, func in funcs.items():
                records.append({
                    "real_season": r_s,
                    "sim_season": s_s,
                    "episode": episode,
                    "metric": func_name,
                    "distance_type": "raw",
                    "distance": calc_distance(sim_series, real_season_series, func),
                })
                records.append({
                    "real_season": r_s,
                    "sim_season": s_s,
                    "episode": episode,
                    "metric": func_name,
                    "distance_type": "shape",
                    "distance": shape_distance(sim_series, real_season_series, func),
                })
                records.append({
                                "real_season": r_s,
                                "sim_season": s_s,
                                "episode": episode,
                                "metric": func_name,
                                "distance_type": "centred",
                                "distance": shape_centred(sim_series, real_season_series, func),
                                })
                records.append({
                                "real_season": r_s,
                                "sim_season": s_s,
                                "episode": episode,
                                "metric": func_name,
                                "distance_type": "mod_shape",
                                "distance": shape_mod_distance(sim_series, real_season_series, func),
                                })

    return pd.DataFrame(records)




        
#%%

def plot_distribution_of_distances_faceted(dist_dist_df, raw_or_shape = "raw", faceting_dim = "real_season ~ metric"):
    plot_df = dist_dist_df[dist_dist_df["distance_type"] == raw_or_shape]
    if raw_or_shape == "raw":
        title = "Raw Distance"
    elif raw_or_shape == "shape":
        title = "Standardised Distance"
    elif raw_or_shape == "centred":
        title = "Centred distance"
    elif raw_or_shape == "mod_shape":
        title = "Standardised (modified) Distance"
    # title = "Raw Distance" if raw_or_shape == "raw" else "Normalised Distance"

    p = (
        ggplot(plot_df, aes(x="distance", fill="factor(sim_season)"))
        + geom_density(alpha=0.5)
        + facet_grid(faceting_dim, scales="free")
        + theme_bw(base_size = 18)
        + theme(figure_size = (14, 9), legend_position = "bottom")
        + labs(fill = "Simulated Season", title = title)
    )
    return p


faceting_dim = "real_season ~ metric"
# faceting_dim = "real_season"
# faceting_dim = "metric"



#%%

def bootstrap_distance(dist_df, n_boot, real_seasons, sim_seasons, funcs,
                        raw_or_shape="raw", funcs_summary={"Mean": np.mean, "Median": np.median},
                        seed=None):
    rng = np.random.default_rng(seed)

    filtered = dist_df[
        (dist_df["distance_type"] == raw_or_shape)
        & (dist_df["real_season"].isin(real_seasons))
        & (dist_df["sim_season"].isin(sim_seasons))
        & (dist_df["metric"].isin(funcs.keys()))
    ]

    # one row per (real_season, episode), one column per (sim_season, metric)
    wide = filtered.pivot_table(
        index=["real_season", "episode"],
        columns=["sim_season", "metric"],
        values="distance"
    )

    unique_real_seasons = wide.index.get_level_values("real_season").unique().to_numpy()
    unique_episodes = wide.index.get_level_values("episode").unique().to_numpy()

    boot_frames = []

    for boot in range(n_boot):
        # sampled_real = rng.choice(unique_real_seasons, size=len(unique_real_seasons), replace=True)
        sampled_episodes = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)

        # cartesian product of the two resampled arrays — vectorized, no python loop
        query_index = pd.MultiIndex.from_product(
            [real_seasons, sampled_episodes], names=["real_season", "episode"]
        )

        # gather in one shot; duplicate keys just repeat the corresponding row,
        # which is exactly what "resample with replacement" means here
        sliced = wide.loc[query_index]

        long = (
            sliced
            .stack(["sim_season", "metric"])  # add future_stack=True if on pandas >= 2.1 to silence the warning
            .rename("value")
            .reset_index()
        )
        long["bootstrap"] = boot
        boot_frames.append(long)

    boot_df = pd.concat(boot_frames, ignore_index=True)
    boot_df = boot_df.rename(columns={"metric": "dist_metric", "sim_season": "season"})

    return boot_df




#%%

if __name__ == "__main__":
    sim_data = pd.read_csv("../generated_data/episode_data.csv")
    real_data = pd.read_csv("../../initial_code/nba_team_historical_percentiles.csv")
    real_data_cleaned = clean_real_data(real_data)

    # real_seasons = ["2012-13" ,"2017-18", "2025-26"]
    real_seasons = [f"20{i}-{i + 1}" for i in range(11, 22)]
    sim_seasons = [0, 9]
    funcs = {
            "Wasserstein" : wasserstein_distance,
            "Energy distance" : energy_distance,
            "KS" : ks_2samp
            }
    


    p = plot_aggregate_kdes_vs_seasons(sim_data, real_data_cleaned, 
                                       real_seasons = real_seasons, 
                                       sim_seasons = sim_seasons)
    p.show()
    
    results_df = distance_between_aggregate_kdes(sim_data, 
                                    real_data_cleaned, 
                                    real_seasons = real_seasons, 
                                    sim_seasons = sim_seasons,
                                    funcs = funcs)

    print("All results:")
    display(results_df)

    dist_dist_df = distribution_of_distance(sim_data, real_data_cleaned, real_seasons, sim_seasons, funcs)
    
    plot_distribution_of_distances_faceted(dist_dist_df, raw_or_shape = "raw", faceting_dim = faceting_dim)
    plot_distribution_of_distances_faceted(dist_dist_df, raw_or_shape = "shape", faceting_dim = faceting_dim)

    

    boot_df = bootstrap_distance(
        dist_df=dist_dist_df,
        n_boot=500,
        real_seasons=real_seasons,
        sim_seasons=sim_seasons,
        funcs=funcs,
        seed=42,
        raw_or_shape="shape"
    )

    
    summary_df = (
        boot_df
        .groupby(["bootstrap", "real_season", "season", "dist_metric"])["value"]
        .agg(mean="mean", median="median")
        .reset_index()
    )

    display(summary_df)

    p_summary = (
        ggplot(summary_df, aes(x = "mean", fill = "factor(season)"))
        + geom_density(alpha = 0.5)
        + facet_grid("real_season ~ dist_metric", scales="free")
        + theme_bw(base_size = 18)
        + theme(figure_size=(20, 12), legend_position="bottom")
    )

    p_summary.show()



    

