import pandas as pd
import numpy as np

def elo_change_calc(winner_elo, loser_elo, k=30):
    """
    Updates the Elo rating of a team based on the result of a game.
    """
    expected_win = 1 / (1 + 10**((loser_elo - winner_elo) / 400))
    elo_change = k * (1 - expected_win)
    return elo_change

def get_initial_elos(teams, initial_rating=1500):
    """
    Creates a dictionary of initial Elo ratings for all teams.
    """
    return {team: initial_rating for team in teams}

def calculate_elo_history(df, k=30, initial_rating=1500):
    """
    Iterates through the DataFrame, calculates Elo changes, and returns the DataFrame 
    with 'home_elo' and 'away_elo' columns added.
    """
    # Get all unique teams
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    current_elos = get_initial_elos(all_teams, initial_rating)
    
    # Store for saving in df later
    home_elo_history = []
    away_elo_history = []
    
    # Iterate through games
    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Get current ratings
        home_rating = current_elos[home]
        away_rating = current_elos[away]
        
        # Store them before the math (so we know what the rating was GOING into the game)
        home_elo_history.append(home_rating)
        away_elo_history.append(away_rating)
        
        # Calculate result
        if row['home_score'] > row['away_score']: # home win
            shift = elo_change_calc(winner_elo=home_rating, loser_elo=away_rating, k=k)
            current_elos[home] += shift
            current_elos[away] -= shift
            
        elif row['away_score'] > row['home_score']: # away win
            shift = elo_change_calc(winner_elo=away_rating, loser_elo=home_rating, k=k)
            current_elos[away] += shift
            current_elos[home] -= shift
            
        else: # tie, no change
            pass 
            
    # Add results to DataFrame
    df_result = df.copy()
    df_result['home_elo'] = home_elo_history
    df_result['away_elo'] = away_elo_history
    
    return df_result

def create_team_elo_chart(df):
    """
    Transforms the DataFrame into the wide format used for plotting.
    Handles Bye Weeks automatically by forward-filling the previous rating.
    """
    long_format = pd.melt(df, 
                          id_vars=['season', 'week', 'home_team', 'away_team'], 
                          value_vars=['home_elo', 'away_elo'],
                          var_name='is_home', value_name='elo')

    long_format['team'] = np.where(
        long_format['is_home'] == 'home_elo', # Condition: Is this a Home Elo row?
        long_format['home_team'],             # If Yes: Use Home Name
        long_format['away_team']              # If No: Use Away Name
    )

    # Pivot to create the chart
    team_elo_chart = long_format.pivot_table(
        index=['season', 'week'], 
        columns='team', 
        values='elo'
    )

    # Fill Bye Weeks
    team_elo_chart = team_elo_chart.ffill()
    
    return team_elo_chart

# Basic ELO win probability calculation
def calc_elo_win(A, B):
    awin = 1 / (1 + 10**( (B - A) / 400))
    return(awin)

# differences in team strength are typically more apparent in the tournament, 
# and the model accounts for this, too. An additional multiplier of 1.07x is applied 
# to the Elo ratings difference between the teams in forecasting margins of 
# victory and win probabilities in the tournament.
# this was for NCAA march madness, needs to be tested for NFL

def calc_elo_win_tourney(A, B, boost=1.07):
    awin = 1 / (1 + 10**( (B - A) * boost / 400))
    return(awin)