#%%
from nhlpy import NHLClient
import pandas as pd
import os

print(os.getcwd())

#%%
client = NHLClient()

def make_season_string(season, for_api = True):
    next_season = season + 1
    if for_api:
        return str(season)+str(next_season)
    else:
        short_season_name = str(next_season)[-2:]
        return f"{season}-{short_season_name}"

def parse_season_data(standings, season):
    records = []

    for entry in standings["standings"]:
        # print(entry)
        gp = entry["gamesPlayed"]
        w_pct_entry = entry["winPctg"]
        team = entry["teamName"]["default"]

        records.append({
            "team" : team,
            "win_pct" : w_pct_entry,
            "gp" : gp,
            "season" : make_season_string(season, for_api=False)
        })

    return pd.DataFrame(records)


all_seasons = []
for season in range(2006, 2026):
    print(f"Downloading season {season}")
    season_string = make_season_string(season, for_api = True)
    standings = client.standings.league_standings(season = season_string)
    season_data = parse_season_data(standings, season)
    all_seasons.append(season_data)

all_seasons = pd.concat(all_seasons)

all_seasons.to_csv("nhl_data.csv", index = False)


