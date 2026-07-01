#%%

import numpy as np
import pandas as pd
from scipy import stats
from plotnine import *
import random

#%%

a = stats.lognorm.rvs(loc = 0, s = 1, size = 1000)
b = stats.lognorm.rvs(loc = 0, s = 0.75, size = 1000)

df = pd.DataFrame({"s1" : a, "s2" : b})
df = df.melt(value_vars=["s1", "s2"])

(
    ggplot(df, aes(x = "value", fill = "variable")) +
    geom_histogram(bins = 100, position = "identity", alpha = 0.7)
)
#%%
N_PLAYERS = 300
N_TEAMS = 30

# player_ratings = truncated_power_law(N_PLAYERS, alpha = 2, xmin = 5.0)
player_ratings = stats.lognorm.rvs(loc = 0, s = 1, size = N_PLAYERS)

p = (
    ggplot(aes(x = player_ratings)) +
    geom_density(color = "black", size = 1, fill = "red", alpha = 0.6)
)
p.show()

print(f"Min: {np.min(player_ratings)}")
print(f"Max: {np.max(player_ratings)}")
print(f"Median: {np.median(player_ratings)}")
print(f"Mean: {np.mean(player_ratings)}")
print(f"Percentiles: {np.percentile(player_ratings, [50, 75, 90, 95, 99])}")
#%%
# Partitioning players between teams
def split_list_in_n(l, cols):
    """ Split up a list in n lists evenly size chunks """
    start = 0
    for i in range(cols):
        stop = start + len(l[i::cols])
        yield l[start:stop]
        start = stop


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

# def assign_players_with_roster_limits(player_ratings, n_teams=30, roster_size=15):
#     """
#     Assigns players to teams while respecting a roster cap.
#     """
#     n_players = len(player_ratings)
#     # Sort players: highest rated first (top talent picks first)
#     sorted_players = np.sort(player_ratings)[::-1]
    
#     # Track remaining spots on each team
#     rosters = [[] for _ in range(n_teams)]
#     spots_remaining = [roster_size] * n_teams
    
#     # Assign players
#     for player in sorted_players:
#         # Calculate current team "Prestige" based on cumulative talent 
#         # (Rich get richer) + a baseline to keep teams viable
#         current_talent = np.array([np.sum(r) for r in rosters])
        
#         # We only consider teams that still have roster spots
#         available_teams = [i for i, spots in enumerate(spots_remaining) if spots > 0]
        
#         # Probability weights: 
#         # Teams with more talent are more attractive (Prestige)
#         weights = current_talent[available_teams] + 1.0  # +1 to handle empty teams
#         probs = weights / np.sum(weights)
        
#         # Select a team
#         target_idx = np.random.choice(available_teams, p=probs)
        
#         rosters[target_idx].append(player)
#         spots_remaining[target_idx] -= 1
        
#     return [r for r in rosters]

teams = list(split_list_in_n(player_ratings, 30))
team_ratings = np.sum(teams, axis = 1)
# team_ratings = np.sum(assign_players_with_roster_limits(player_ratings, n_teams = N_TEAMS, roster_size = N_PLAYERS / N_TEAMS), axis = 1)
p = qplot(team_ratings, geom = "density")
p.show()

#%%
N_MATCH = 1000
rng = np.random.default_rng()
diff_matrix = team_ratings[:, None] - team_ratings[None, :]
diff_matrix = np.repeat(diff_matrix[:, :, np.newaxis], N_MATCH, axis=2)
noise = rng.logistic(loc=0.0, scale=1, size=(N_TEAMS, N_TEAMS, N_MATCH))
outcomes = (diff_matrix + noise) > 0
# np.fill_diagonal(outcomes, False)
idx = np.arange(N_TEAMS) # Sets diagonal to False
outcomes[idx, idx, :] = False

wins = outcomes.sum(axis=(1, 2))
losses = (N_TEAMS - 1) * N_MATCH - wins

print(wins / ((N_TEAMS - 1) * N_MATCH))
print(f"Gini = {calculate_gini(wins / (N_TEAMS - 1))}")


#%%
def generate_nba_schedule(n_teams=30):
    # This creates a list of 82 opponents for every team
    # team_indices: list of 0 to 29
    # Divisions: Assume 0-4 are Div1, 5-9 Div2, etc. (6 divisions of 5)
    
    schedules = {i: [] for i in range(n_teams)}
    
    for i in range(n_teams):
        # 1. Division: 4 rivals, play 4 times each
        division_mates = [t for t in range((i//5)*5, (i//5)*5 + 5) if t != i]
        schedules[i].extend(np.repeat(division_mates, 4))
        
        # 2. Conference (Intra-Conference, Non-Division): 10 teams
        # Simplified: Play these 10 teams ~3.6 times each
        conference_mates = [t for t in range((i//15)*15, (i//15)*15 + 15) if t not in division_mates and t != i]
        schedules[i].extend(np.random.choice(conference_mates, size=36, replace=True))
        
        # 3. Non-Conference: 15 teams
        other_conf = [t for t in range(n_teams) if t not in range((i//15)*15, (i//15)*15 + 15)]
        schedules[i].extend(np.repeat(other_conf, 2))

        # random.shuffle(schedules[i])
        
    return schedules

schedules = generate_nba_schedule( n_teams = N_TEAMS)

#%%

def generate_exact_nba_schedule(n_teams=30):
    game_list = []
    
    # Loop through every unique pair of teams exactly once
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            
            # Determine relationships (assuming teams 0-14 are Conf 1, 15-29 are Conf 2)
            # Divisions are chunks of 5 (e.g., 0-4, 5-9, etc.)
            conf_i, conf_j = i // 15, j // 15
            div_i, div_j = i // 5, j // 5
            
            if conf_i != conf_j:
                # 1. Non-Conference: 2 games
                num_games = 2
                
            elif div_i == div_j:
                # 2. Same Division: 4 games
                num_games = 4
                
            else:
                # 3. Same Conference, Different Division: 3 or 4 games
                # We need exactly 6 matchups to have 4 games, and 4 matchups to have 3 games.
                # This mathematical trick evenly and symmetrically distributes the 3s and 4s 
                # across all teams in the conference.
                if (i + j) % 5 < 3:
                    num_games = 4
                else:
                    num_games = 3
                    
            # Append this matchup the required number of times
            for _ in range(num_games):
                game_list.append((i, j))
    
    game_list = random.sample(game_list, len(game_list)) # Shuffle game sequence (sample w/o replacement)

    print(game_list)
    return game_list

# Generate the global list
g_list = generate_exact_nba_schedule()

def play_season(team_ratings, game_list, noise_scale, pct_played = 1):
    wins = np.zeros(len(team_ratings))
    games_played = np.zeros(len(team_ratings)) # Track total games per team

    n_to_play = round(len(game_list) * pct_played)
    
    for game_id in range(n_to_play):
        game = game_list[game_id]
        team_a = game[0]
        team_b = game[1]
        # Calculate latent difference
        diff = team_ratings[team_a] - team_ratings[team_b]
        # Generate one shared random noise for this specific game
        noise = np.random.logistic(loc=0.0, scale=noise_scale)
        
        if (diff + noise) > 0:
            wins[team_a] += 1
        else:
            wins[team_b] += 1
            
        # Log that both teams played a game
        games_played[team_a] += 1
        games_played[team_b] += 1
            
    return wins, games_played

# Run the simulation and unpack both variables
wins, games_played = play_season(team_ratings, g_list, noise_scale=10, pct_played=1)
print(f"{np.mean(games_played)} is the average of games played until this point in the season")
print(games_played)
print(len(games_played))
win_pct = wins / games_played

# Divide wins by the exact number of games each team played
print(f"Win %:\n{win_pct}")
print(f"Gini: {calculate_gini(win_pct)}")


#%%
import itertools
N_SEASONS = 16
win_pct_list = []
for s in range(N_SEASONS):
    print(s)
    player_ratings = stats.lognorm.rvs(loc = 0, s = 1, size = N_PLAYERS)
    # player_ratings = sample_power_law(N_PLAYERS, alpha = 4, x_min = 5)
    teams = list(split_list_in_n(player_ratings, 30))
    team_ratings = np.sum(teams, axis = 1)
    g_list = generate_exact_nba_schedule()

    wins, games_played = play_season(team_ratings, g_list, noise_scale=10)
    win_pct = wins / games_played
    win_pct_list.append(win_pct)

win_pct_list = list(itertools.chain.from_iterable(win_pct_list)) # From list of lists to one list


#%%
final_teams_df = pd.read_csv("nba_team_historical_percentiles.csv")

final_teams_df
#%%
win_pct_df = pd.DataFrame({"Simulation" : win_pct_list, "Real" : final_teams_df["WinPCT"]})

win_pct_df = win_pct_df.melt(value_vars = ["Simulation", "Real"])
(
    ggplot(win_pct_df, aes(x = "value", fill = "variable")) +
    geom_density(size = 1, alpha = 0.6, color = "black") +
    labs(x = "Win %")
)

#%%
p_wrap = (
    ggplot(final_teams_df, aes(x = "WinPCT", fill = "Season", color = "Season")) +
    geom_density(alpha = 0.6) +
    facet_wrap("~Season") 
)
p_wrap.show()

p_all = (
    ggplot(final_teams_df, aes(x = "WinPCT", color = "Season")) +
    geom_density() 
)
p_all.show()

#%%
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
    # Fetch historical data (2010-11 to 2025-26)
    raw_df = fetch_nba_historical_data(start_year=2010, end_year=2025)
    
    # Process calculations
    final_teams_df, gini_df = process_advanced_standings_metrics(raw_df)
    
    # Sort teams by season and performance
    final_teams_df = final_teams_df.sort_values(by=['Season', 'WinPCT'], ascending=[False, False])
    
    # Clean visual column names for output
    final_teams_df['Win %'] = (final_teams_df['WinPCT'] * 100).round(1).astype(str) + '%'
    
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

