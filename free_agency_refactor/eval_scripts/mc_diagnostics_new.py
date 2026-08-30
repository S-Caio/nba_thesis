#%%
import pandas as pd
import numpy as np
from datetime import datetime
from plotnine import *
from scipy.stats import wasserstein_distance, energy_distance, ks_2samp
from mc_diag_utils import (clean_real_data, 
                           plot_aggregate_kdes_vs_seasons,
                            distance_between_aggregate_kdes,
                            distribution_of_distance,
                            plot_distribution_of_distances_faceted,
                            bootstrap_distance)


#%%

def main():
    sim_data = pd.read_csv("../generated_data/episode_data.csv")
    real_data = pd.read_csv("../../initial_code/nba_team_historical_percentiles.csv")
    real_data_cleaned = clean_real_data(real_data)

    today = datetime.today().strftime("%d_%m_%y")

    plot_suffix = "small"
    if plot_suffix == "small":
        real_seasons = ["2012-13" ,"2017-18", "2020-21", "2025-26"]
    elif plot_suffix == "large":
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
    p.save(f"generated_plots/season_kdes_{today}_{plot_suffix}.pdf")

    
    results_df = distance_between_aggregate_kdes(sim_data, 
                                    real_data_cleaned, 
                                    real_seasons = real_seasons, 
                                    sim_seasons = sim_seasons,
                                    funcs = funcs)

    print("All results:")
    col_names = []
    for func_name in funcs.keys():
        col_names.append(f"{func_name}_raw")
        col_names.append(f"{func_name}_shape")
        col_names.append(f"{func_name}_centred")
    display(results_df[["real_season", "sim_season"] + col_names])
    print(results_df[["real_season", "sim_season"] + col_names].to_latex(float_format="%.4f", index = False))
    # display(results_df)

    dist_dist_df = distribution_of_distance(sim_data, real_data_cleaned, real_seasons, sim_seasons, funcs)

    faceting_dim = "real_season ~ metric"
    p_raw = plot_distribution_of_distances_faceted(dist_dist_df, raw_or_shape = "raw", faceting_dim = faceting_dim)
    p_shape = plot_distribution_of_distances_faceted(dist_dist_df, raw_or_shape = "shape", faceting_dim = faceting_dim)
    p_centred = plot_distribution_of_distances_faceted(dist_dist_df, raw_or_shape = "centred", faceting_dim = faceting_dim)
    p_shape_mod = plot_distribution_of_distances_faceted(dist_dist_df, raw_or_shape = "mod_shape", faceting_dim = faceting_dim)


    p_raw.show()
    p_shape.show()
    p_centred.show()
    p_shape_mod.show()

    p_raw.save(f"generated_plots/RAW_distribution_of_distances_{today}_{plot_suffix}.pdf")
    p_shape.save(f"generated_plots/SHAPE_distribution_of_distances_{today}_{plot_suffix}.pdf")
    p_centred.save(f"generated_plots/CENTRED_distribution_of_distances_{today}_{plot_suffix}.pdf")
    p_shape_mod.save(f"generated_plots/SHAPE_MOD_distribution_of_distances_{today}_{plot_suffix}.pdf")

    boot_df = bootstrap_distance(
        dist_df=dist_dist_df,
        n_boot=500,
        real_seasons=real_seasons,
        sim_seasons=sim_seasons,
        funcs=funcs,
        seed=42,
        raw_or_shape="mod_shape"
    )

    
    summary_df = (
        boot_df
        .groupby(["bootstrap", "real_season", "season", "dist_metric"])["value"]
        .agg(mean="mean", median="median")
        .reset_index()
    )

    display(summary_df)

    estim = "mean"
    p_summary = (
        ggplot(summary_df, aes(x = estim, fill = "factor(season)"))
        + geom_density(alpha = 0.5)
        + facet_grid("real_season ~ dist_metric", scales="free")
        + theme_bw(base_size = 18)
        + theme(figure_size=(20, 12), legend_position="bottom")
        + labs(title = "Bootstrapped mean estimator of distance" if estim == "mean" else "Bootstrapped median estimator of distance",
               fill = "Simulated season")
    )

    p_summary.show()
    # p_summary.save(f"generated_plots/bootstrapped_{estim}_dist_{today}_{plot_suffix}.pdf")



if __name__ == "__main__":
    main()