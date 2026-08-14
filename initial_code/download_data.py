import time
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguestandings

def calculate_gini(array):
    """
    Calculates the Gini coefficient of a numpy array.
    A higher Gini coefficient implies greater competitive inequality (more tanking/superteams).
    A lower Gini implies higher league parity (everyone close to .500).
    """
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        return None
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)

def fetch_nba_historical_data(start_year=2010, end_year=2025):
    """
    Fetches standings data from start_year (e.g., 2010 for '2010-11')
    to end_year (e.g., 2025 for '2025-26').
    """
    all_seasons_list = []

    # Generate season strings (e.g., 2010-11, 2011-12... 2025-26)
    seasons = [f"{year}-{str(year+1)[2:]}" for year in range(start_year, end_year + 1)]

    for season in seasons:
        print(f"Fetching standings data for {season}...")
        try:
            standings = leaguestandings.LeagueStandings(
                league_id='00',
                season=season,
                season_type='Regular Season'
            )
            df = standings.get_data_frames()[0]

            # Construct Team Name and attach season identifier
            df['Team'] = df['TeamCity'] + ' ' + df['TeamName']
            df['Season'] = season

            # Extract core columns
            df_filtered = df[['Season', 'Team', 'WINS', 'LOSSES', 'WinPCT']].copy()
            all_seasons_list.append(df_filtered)

            # Sleep to protect against NBA API rate limits
            time.sleep(1.5)

        except Exception as e:
            print(f"⚠️ Error fetching {season}: {e}. Retrying after a short pause...")
            time.sleep(5)

    if all_seasons_list:
        return pd.concat(all_seasons_list, ignore_index=True)
    else:
        raise Exception("No data could be retrieved from the NBA API.")

def process_advanced_standings_metrics(df):
    # 1. Calculate within-season win percentage percentile
    # (Rank of team within its specific season, 100% being the best team that year)
    df['Percentile_Within_Season'] = (
        df.groupby('Season')['WinPCT'].rank(pct=True) * 100
    ).round(1)

    # 2. Calculate across-all-seasons win percentage percentile
    # (Rank of team compared to EVERY team/season since 2010)
    df['Percentile_Across_All_Seasons'] = (
        df['WinPCT'].rank(pct=True) * 100
    ).round(1)

    # 3. Calculate the within-season Gini Coefficient
    gini_records = []
    for season, group in df.groupby('Season'):
        gini_val = calculate_gini(group['WinPCT'].values)
        gini_records.append({'Season': season, 'Gini_Coefficient': round(gini_val, 4)})

    df_gini = pd.DataFrame(gini_records)

    return df, df_gini

if __name__ == "__main__":
    START_YEAR = 2010
    END_YEAR = 2025
    # Fetch historical data (2010-11 to 2025-26)
    raw_df = fetch_nba_historical_data(start_year=START_YEAR, end_year=END_YEAR)

    # Process calculations
    final_teams_df, gini_df = process_advanced_standings_metrics(raw_df)

    # Sort teams by season and performance
    final_teams_df = final_teams_df.sort_values(by=['Season', 'WinPCT'], ascending=[False, False])

    # Clean visual column names for output
    final_teams_df['Win %'] = (final_teams_df['WinPCT'] * 100).round(1).astype(str) + '%'

    # Add system label
    seasons_current_system = [f"20{i}-{i + 1}" for i in range(19, 27)]
    final_teams_df["system"] = np.where(final_teams_df["Season"].isin(seasons_current_system), "Current", "Old")

    # Save datasets to CSV files
    final_teams_df.to_csv('nba_team_historical_percentiles.csv', index=False)
    gini_df.to_csv('nba_season_gini_coefficients.csv', index=False)

    # --- REPORTING DATA ---
    print("\n========================================================")
    print("🏀 REPORT 1: WITHIN-SEASON COMPETITIVE INEQUALITY (GINI)")
    print("========================================================")
    print("Note: A higher Gini means less balance (clear superteams/tankers).")
    print(gini_df.to_string(index=False))

    print("\n========================================================")
    print("🏀 REPORT 2: TOP 10 INDIVIDUAL TEAM SEASONS (HISTORICAL PERCENTILE)")
    print("========================================================")
    top_10 = final_teams_df.sort_values(by='Percentile_Across_All_Seasons', ascending=False).head(10)
    print(top_10[['Season', 'Team', 'Win %', 'Percentile_Across_All_Seasons']].to_string(index=False))

    print("\n========================================================")
    print("🏀 REPORT 3: SAMPLE TEAM EVALUATION (e.g., 2023-24 Season Top 5)")
    print("========================================================")
    sample_season = final_teams_df[final_teams_df['Season'] == '2023-24'].head(5)
    print(sample_season[['Season', 'Team', 'Win %', 'Percentile_Within_Season', 'Percentile_Across_All_Seasons']].to_string(index=False))

    print("\n📊 Saved full data tables to 'nba_team_historical_percentiles.csv' and 'nba_season_gini_coefficients.csv'")

