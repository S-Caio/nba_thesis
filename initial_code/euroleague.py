#%%
from euroleague_api.standings import Standings
from euroleague_api.schedule import Schedule
import numpy as np
import pandas as pd

#%%
def get_num_games(season):
    sche = Schedule().get_schedule(season)
    max_num_games = sche[sche["round"] == "RS"]["gameday"].max()
    return max_num_games

def turn_pct_into_num(text):
    num = text.split("%")[0]
    return float(num) / 100

def get_and_clean_data(season, games, season_year):
    st = Standings().get_standings(season, round_number=games)
    st = st[["club.abbreviatedName", "winPercentage", "gamesPlayed"]]
    st = st.rename(
        {"club.abbreviatedName" : "team",
        "winPercentage" : "win_pct",
        "gamesPlayed" : "GP"
        }, 
        axis = 1)

    st["win_pct"] = st["win_pct"].apply(turn_pct_into_num)
    st["season"] = season_year
    return st


SEASONS = np.arange(2016, 2026)
records = []


for starting_season in SEASONS:
    print(starting_season)
    num_games = get_num_games(starting_season)
    data = get_and_clean_data(starting_season, num_games, starting_season)
    records.append(data)

df = pd.concat(records)

df.to_csv("euroleague_data.csv", index = False)
print("Scraped and saved EuroLeague data!")