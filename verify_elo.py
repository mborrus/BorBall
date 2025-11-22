import pandas as pd
import numpy as np
from elo_functions import calculate_elo_history, create_team_elo_chart

# Load data
try:
    # Try loading the file we downloaded earlier
    df = pd.read_csv("games.csv")
    
    # Filter for completed games like in the notebook
    # The notebook used nfl_data_py, but games.csv has similar structure
    # Let's check columns. games.csv usually has 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score'
    
    # Minimal prep to match notebook expectations
    # Notebook: elo_prep = completed_games[['game_id','season','game_type','gameday', 'week','away_team','home_team','away_score','home_score']]
    
    # We need to ensure we have these columns and no NaNs in scores
    completed_games = df[df['result'].notna()].copy()
    
    # Map teams if needed (notebook had a mapping function)
    def map_team_abbreviations(df):
        team_map = {
            'STL': 'LA',   # St. Louis Rams -> LA Rams
            'SD': 'LAC',   # San Diego Chargers -> LA Chargers
            'OAK': 'LV'    # Oakland Raiders -> Las Vegas Raiders
        }
        df['home_team'] = df['home_team'].replace(team_map)
        df['away_team'] = df['away_team'].replace(team_map)
        return df

    completed_games = map_team_abbreviations(completed_games)
    
    elo_prep = completed_games[['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    
    print("Data loaded and prepared.")
    
    # 1. Calculate Elo History
    print("Calculating Elo history...")
    elo_prep = calculate_elo_history(elo_prep)
    print("Elo history calculated.")
    print(elo_prep[['home_team', 'home_elo', 'away_team', 'away_elo']].head())
    
    # 2. Create Chart Data
    print("Creating chart data...")
    team_elo_chart = create_team_elo_chart(elo_prep)
    print("Chart data created.")
    print(team_elo_chart.iloc[:5, :5])
    
    print("Verification successful!")
    
except Exception as e:
    print(f"Verification failed: {e}")
