#%%
import pandas as pd
import numpy as np
from plotnine import *
from which_dist_func_utils import shape_distance, clean_real_data
from scipy.stats import energy_distance
from itertools import product, pairwise, combinations

#%%
sim_df = pd.read_csv("../generated_data/episode_data.csv")
real_df = clean_real_data(pd.read_csv("../../initial_code/nba_team_historical_percentiles.csv"))

def distance_between_seasons(seasons, df, season_name_var = "real_season", dist_func = energy_distance):
    """
    Input: Collection of seasons
    Output: Distribution of distances between seasons
    """
    dists = []

    for i, j in combinations(seasons, 2):
        season_i = df[df[season_name_var] == i]["win_pct"]
        season_j = df[df[season_name_var] == j]["win_pct"]

        d = shape_distance(season_i, season_j, distance_func = dist_func)
        dists.append({
            "season_i" : i,
            "season_j" : j,
            "distance" : d
        }
        )

    return dists




real_df["system"] = ["Current" if int(i[-2:]) >= 19 else "Old" for i in real_df["real_season"]]
current_seasons = list(real_df[real_df["system"] == "Current"]["real_season"].unique())
old_seasons = list(real_df[real_df["system"] != "Current"]["real_season"].unique())


#%%
# Testing similarity within eras to see if there is any difference

records = []
seasons_dict = {"Old" : old_seasons, "Current" : current_seasons}
for system_name, system_seasons in seasons_dict.items():
    distances = distance_between_seasons(system_seasons, real_df)
    data = pd.DataFrame(distances)
    data["system"] = system_name
    
    records.append(data)

dist_df = pd.concat(records)

p_box = (
    ggplot(dist_df, aes(x = "distance", fill = "system", color = "system"))
    + geom_histogram(aes(y=after_stat("density")), alpha=0.3, position="identity")
)

p_box.show()

p_jitter = (
    ggplot(dist_df, aes(x = "system", y = "distance", color = "system"))
    + geom_jitter(width = 0.15, height = 0, size = 2.5)
    + labs(x = "", y = "Energy Distance", title = "Distance Within Eras")
    + theme(legend_position = "none")
)

p_jitter.show()
# # distance_between_seasons(["2012-13", "2013-14", "2014-15"], real_df)


#%%
# More general: Now disregarding era/system, I just want to see the overall distribution
from scipy.stats import wasserstein_distance

records = []
seasons_iter = old_seasons + current_seasons
dists = distance_between_seasons(seasons_iter, real_df)
dists_df_all = pd.DataFrame(dists)

p_overall = (
    ggplot(dists_df_all, aes(x = "distance"))
    + geom_histogram(aes(y = after_stat("density")), alpha = 0.3, position = "identity", fill = "blue")
    + geom_density(color = "blue")
)

old_old = (dists_df_all["season_i"].isin(old_seasons)) & (dists_df_all["season_j"].isin(old_seasons))
curr_curr = (dists_df_all["season_i"].isin(current_seasons)) & (dists_df_all["season_j"].isin(current_seasons))
mixed = ~old_old & ~curr_curr  # anything not purely old or purely current

dists_df_all["comb_type"] = np.select(
    [old_old, curr_curr, mixed],
    ["Same era (old)", "Same era (current)", "Cross-era"],
    default="Unclassified"
)


# --- Overlaid densities ---
p_density = (
    ggplot(dists_df_all, aes(x = "distance", fill = "comb_type", color = "comb_type"))
    + geom_density(alpha = 0.3)
    + labs(x = "Energy Distance", y = "Density", 
           title = "Distribution of Season Distances", subtitle = "By Era Combination",
           fill = "Combination", color = "Combination")
)

# --- Violin plot ---
p_violin = (
    ggplot(dists_df_all, aes(x = "comb_type", y = "distance", fill = "comb_type"))
    + geom_violin(alpha = 0.5, trim = False)
    + geom_jitter(width = 0.1, height = 0, alpha = 0.4, size = 1.5)
    + labs(x = "", y = "Energy Distance", 
           title = "Pairwise Season Distances by Era Combination")
    + theme(legend_position = "none")
)

p_density.show()
p_violin.show()

dists_df_all["era_type"] = np.where(
    dists_df_all["comb_type"] == "Cross-era", "Cross-era", "Same era"
)

p_era_density = (
    ggplot(dists_df_all, aes(x = "distance", fill = "era_type", color = "era_type"))
    + geom_density(alpha = 0.3)
    + labs(x = "Energy Distance", y = "Density",
           title = "Same-Era vs Cross-Era Season Distances",
           fill = "", color = "")
)

p_era_violin = (
    ggplot(dists_df_all, aes(x = "era_type", y = "distance", fill = "era_type"))
    + geom_violin(alpha = 0.5, trim = False)
    + geom_jitter(width = 0.1, height = 0, alpha = 0.4, size = 1.5)
    + labs(x = "", y = "Energy Distance", title = "Same-Era vs Cross-Era Season Distances")
    + theme(legend_position = "none")
)

p_era_density.show()
p_era_violin.show()

#%%
# def bootstrap_group_mean(seasons, df, n_boot=1000, season_name_var="real_season", dist_func=energy_distance):
#     boot_means = []
#     for _ in range(n_boot):
#         sample = np.random.choice(seasons, size=len(seasons), replace=True)
#         pair_dists = []
#         for i, j in combinations(range(len(sample)), 2):
#             s_i, s_j = sample[i], sample[j]
#             if s_i == s_j:
#                 continue  # skip self-pairs (same season drawn twice)
#             season_i = df[df[season_name_var] == s_i]["win_pct"]
#             season_j = df[df[season_name_var] == s_j]["win_pct"]
#             pair_dists.append(shape_distance(season_i, season_j, distance_func=dist_func))
#         if pair_dists:  # guard against edge case of all-duplicate draws
#             boot_means.append(np.mean(pair_dists))
#     return boot_means

# boot_old = bootstrap_group_mean(old_seasons, real_df, n_boot=1000)
# boot_current = bootstrap_group_mean(current_seasons, real_df, n_boot=1000)
# boot_cross = bootstrap_group_mean(old_seasons + current_seasons, real_df, n_boot=1000) 
# print("Done with bootstrap group mean!")


estim = np.std

def bootstrap_cross_era_mean(old_seasons, current_seasons, df, n_boot=1000, 
                               season_name_var="real_season", dist_func=energy_distance, estimator = np.std):
    boot_means = []
    for _ in range(n_boot):
        old_sample = np.random.choice(old_seasons, size=len(old_seasons), replace=True)
        curr_sample = np.random.choice(current_seasons, size=len(current_seasons), replace=True)
        
        pair_dists = []
        for s_i, s_j in product(old_sample, curr_sample):
            # no need to check s_i == s_j here — old and current season labels never overlap
            season_i = df[df[season_name_var] == s_i]["win_pct"]
            season_j = df[df[season_name_var] == s_j]["win_pct"]
            pair_dists.append(shape_distance(season_i, season_j, distance_func=dist_func))
        
        boot_means.append(estimator(pair_dists))
    return boot_means

boot_cross = bootstrap_cross_era_mean(old_seasons, current_seasons, real_df, n_boot=1000, estimator = estim)

def bootstrap_same_era_mean(old_seasons, current_seasons, df, n_boot=1000,
                              season_name_var="real_season", dist_func=energy_distance, estimator = np.std):
    boot_means = []
    for _ in range(n_boot):
        old_sample = np.random.choice(old_seasons, size=len(old_seasons), replace=True)
        curr_sample = np.random.choice(current_seasons, size=len(current_seasons), replace=True)
        
        pair_dists = []
        for sample in (old_sample, curr_sample):
            for i, j in combinations(range(len(sample)), 2):
                s_i, s_j = sample[i], sample[j]
                if s_i == s_j:
                    continue
                season_i = df[df[season_name_var] == s_i]["win_pct"]
                season_j = df[df[season_name_var] == s_j]["win_pct"]
                pair_dists.append(shape_distance(season_i, season_j, distance_func=dist_func))
        
        if pair_dists:
            boot_means.append(estimator(pair_dists))
    return boot_means

boot_same_era = bootstrap_same_era_mean(old_seasons, current_seasons, real_df, n_boot=1000, estimator = estim)

boot_df = pd.DataFrame({
    "bootstrapped_estimator": boot_same_era + boot_cross,
    "era_type": ["Same era"] * len(boot_same_era) + ["Cross-era"] * len(boot_cross)
})


#%%
estimator_string = "$\\sigma $"

p_boot_density = (
    ggplot(boot_df, aes(x = "bootstrapped_estimator", fill = "era_type", color = "era_type"))
    + geom_density(alpha = 0.3)
    + labs(x = f"Bootstrapped {estimator_string} Distance", y = "Density",
           title = f"Bootstrapped {estimator_string} Distance: Same-Era vs Cross-Era",
           subtitle = f"{len(boot_same_era)} bootstrap iterations per group",
           fill = "", color = "")
)

p_boot_density.show()

# np.mean(boot_same_era < boot_cross)
boot_same_era = np.array(boot_same_era)
boot_cross = np.array(boot_cross)

np.mean(boot_same_era < boot_cross)


#%%
# Permutation tests (for now within the NBA data only)

# Pre-group win percentages by season into NumPy arrays
season_data = {
    season: grp["win_pct"].to_numpy()
    for season, grp in real_df.groupby("real_season")
}
season_names = np.array(list(season_data.keys()))

# Assign era labels consistently based on season ending year (e.g. '2018-19' -> 19)
is_current_era = np.array([int(s[-2:]) >= 19 for s in season_names])


def compute_era_distance(current_mask):
  curr_pcts = np.concatenate(
      [season_data[s] for s in season_names[current_mask]]
  )
  old_pcts = np.concatenate(
      [season_data[s] for s in season_names[~current_mask]]
  )
  return shape_distance(curr_pcts, old_pcts, distance_func=energy_distance)


# Compute observed statistic
obs_value = compute_era_distance(is_current_era)

# Run permutation test
N_PERM = 10_000
perm_dists = np.zeros(N_PERM)

for i in range(N_PERM):
  perm_mask = np.random.permutation(is_current_era)
  perm_dists[i] = compute_era_distance(perm_mask)

# Right-tailed p-value calculation
p_value = (np.sum(perm_dists >= obs_value) + 1) / (N_PERM + 1)

print(f"Observed Energy Distance: {obs_value:.5f}")
print(f"Permutation p-value: {p_value:.4f}")



    






#%%
from diptest import dipstat

def noll_scully_ratio(series):
    isd = 0.5 / (np.sqrt(82))
    return series.std() / isd

def gini_coefficient(x):
    """
    Compute the Gini coefficient of an array of values (e.g., win_pct per team).
    0 = perfect equality, 1 = maximal inequality.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    
    # cumulative sum-based formula (mean absolute difference approach)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))

stats_df = real_df.groupby("real_season")["win_pct"].agg(
    nsr = lambda x: noll_scully_ratio(x),
    sigma = lambda x: np.std(x),
    dipstat = lambda x: dipstat(x),
    gini = lambda x: gini_coefficient(x)
).reset_index()

def plot_stat(stats_df, stat_to_plot, figure_size = (16, 12), title = None, x_lab = None, y_lab = None):
    p = (
        ggplot(stats_df, aes(x = "real_season", y = stat_to_plot, group = 1))
        + geom_point(size = 5)
        + geom_line()
        + theme_bw(base_size = 18)
        + theme(figure_size = figure_size)
        + labs(title = title, x = x_lab, y = y_lab)
    )
    return p


stats_df["real_season"] = pd.Categorical(stats_df["real_season"])
p_nsr = plot_stat(stats_df, "nsr")
p_sigma = plot_stat(stats_df, "sigma")
p_dip = plot_stat(stats_df, "dipstat")
p_gini = plot_stat(stats_df, "gini")


p_sigma / p_dip / p_gini


#%%
def top_bottom_stats(group):
    x = group["win_pct"].sort_values(ascending=False).to_numpy()

    return pd.Series({
        "top5_mean_w_pct": x[:5].mean(),
        "bottom5_mean_w_pct": x[-5:].mean(),
        "top5_bottom5_gap": x[:5].mean() - x[-5:].mean(),
        "top5_sd": x[:5].std(),
        "bottom5_sd": x[-5:].std(),
        "median_w_pct": np.median(x),
        "iqr_w_pct": np.percentile(x, 75) / np.percentile(x, 25),
        "95_05_ratio" : np.percentile(x, 95) / np.percentile(x, 5)
    })

tail_stats_df = (
    real_df
    .groupby("real_season")
    .apply(top_bottom_stats)
    .reset_index()
)

display(tail_stats_df)

for i in tail_stats_df.columns:
    if type(tail_stats_df[i][0]) == str:
        continue

    p = plot_stat(tail_stats_df, stat_to_plot = i,
              title = i)
    p.show()


#%%
# Trying to see if a multidimensional feature vector can cleanly separate eras


dfs = stats_df.merge(tail_stats_df, left_on="real_season", right_on="real_season")
dfs.drop("sigma", axis = 1, inplace = True)
feature_cols = [i for i in dfs.columns if i != "real_season"]

feature_cols

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

stat_feats = dfs[feature_cols]
stat_feats = StandardScaler().fit_transform(stat_feats)
stat_feats = pd.DataFrame(stat_feats, columns = feature_cols)

N_COMPONENTS = 2

pca = PCA(n_components=N_COMPONENTS)
pca_feats = pca.fit_transform(stat_feats)
pca_feats = pd.DataFrame(pca_feats, columns = [f"comp_{i}" for i in range(N_COMPONENTS)])
pca_feats["real_season"] = dfs["real_season"]
pca_feats
pca_feats["system"] = ["Current" if int(i[-2:]) >= 19 else "Old" for i in pca_feats["real_season"]]
(
    ggplot(pca_feats, aes(x = "comp_0", y = "comp_1", color = "system"))
    + geom_point(size = 4, shape = "d")
)

#%%
# Looking at autocorrelation of win percentages
team_mapping = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Jersey Nets": "Brooklyn Nets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "Los Angeles Clippers" : "LA Clippers"
}

real_df["team"] = real_df["team"].replace(team_mapping)

pivot = real_df.pivot_table(values = "win_pct", index = "team", columns = "real_season")

real_df["lag_rank"] = (
    real_df
    .groupby("real_season")["lag_win_pct"]
    .rank(ascending=False, method="average")
)

real_df["current_rank"] = (
    real_df
    .groupby("real_season")["win_pct"]
    .rank(ascending=False, method="average")
)

corr_by_season = (
    real_df
    .groupby("real_season")
    .apply(lambda g: g["current_rank"].corr(g["lag_rank"], method="spearman"))
    .reset_index(name="rho")
)

corr_by_season.dropna(inplace=True)

(
    ggplot(corr_by_season, aes(x = "real_season", y = "rho", group = 1))
    + geom_point()
    + geom_line()
    + theme_bw(base_size=18)
    + theme(figure_size=(16, 8))
    + labs(y = "$\\rho$", x = "Season")
)


# old_season = None
# for season in pivot.columns:
#     if old_season is None:
#         continue

#     pivot[f"season_diff_{season}"] = pivot[season] - pivot[old_season]

#     old_season = season


#%%
from sklearn.mixture import GaussianMixture
from scipy.special import logit
import numpy as np

def bimodality_bic(x, max_components=2):
    x = np.asarray(x).reshape(-1, 1)
    results = {}
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, n_init=10, random_state=0)
        gmm.fit(x)
        results[k] = {
            "bic": gmm.bic(x),
            "aic": gmm.aic(x),
            "loglik": gmm.score(x) * len(x)
        }
    return results

def bimodality_bic_logit(x, max_components=2):
    x = np.clip(np.asarray(x), 1e-4, 1 - 1e-4)  # avoid inf at 0/1
    return bimodality_bic(logit(x), max_components=max_components)

def bic_diff_2v1(x):
    res = bimodality_bic_logit(x)
    return res[1]["bic"] - res[2]["bic"]

def bootstrap_bimodality(x, n_boot=200):
    x = np.asarray(x)
    votes_for_2 = 0
    for _ in range(n_boot):
        sample = np.random.choice(x, size=len(x), replace=True)
        res = bimodality_bic(sample)
        if res[2]["bic"] < res[1]["bic"]:
            votes_for_2 += 1
    return votes_for_2 / n_boot

a = real_df[real_df["real_season"] == "2015-16"]["win_pct"]
b = real_df[real_df["real_season"] == "2025-26"]["win_pct"]

# bootstrap_bimodality(b)
stats_df = real_df.groupby("real_season")["win_pct"].agg(
    bic_diff = lambda x: bic_diff_2v1(x)
).reset_index()

p_bic = plot_stat(stats_df, "bic_diff")
p_bic = (
    p_bic 
    + geom_hline(yintercept = 0)
)

p_bic