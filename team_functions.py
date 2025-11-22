import pandas as pd

def map_team_abbreviations(df):
    df_copy = df.copy()
    team_map = {
        'STL': 'LA',   # St. Louis Rams -> LA Rams
        'SD': 'LAC',   # San Diego Chargers -> LA Chargers
        'OAK': 'LV'    # Oakland Raiders -> Las Vegas Raiders
    }
    df_copy['home_team'] = df_copy['home_team'].replace(team_map)
    df_copy['away_team'] = df_copy['away_team'].replace(team_map)
    return df_copy