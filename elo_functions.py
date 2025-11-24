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

def elo_win_prob(elo_prep, HFA_elo_adjustment=0):
    ''' to be run after you calculate elo history '''
    elo_prep['home_win_prob']=elo_prep.apply(lambda row: calc_elo_win(row['home_elo']+HFA_elo_adjustment, row['away_elo']), axis=1)
    elo_prep['away_win_prob']=1-elo_prep['home_win_prob']
    return elo_prep

def get_elo_metrics(elo_prep):
    ''' to be run after you calculate elo history '''
    min_season = elo_prep['season'].min()
    elo_prep_testing = elo_prep[elo_prep['season'] > min_season + 1]

    win_outcome = elo_prep_testing['home_score'] > elo_prep_testing['away_score']
    win_outcome = win_outcome.astype(int)

    win_predict_bin = elo_prep_testing['home_win_prob'] > 0.5
    win_predict_bin = win_predict_bin.astype(int)

    brier_score = brier_score_loss(win_outcome, elo_prep_testing['home_win_prob'])
    accuracy = accuracy_score(win_outcome, win_predict_bin)
    confusion = confusion_matrix(win_outcome, win_predict_bin)
    return brier_score, accuracy, confusion

def fit_HFA_elo(df, k=35):
    from sklearn.linear_model import LinearRegression
    # 1. Filter for games with results (Drop future games)
    df_reg = df.dropna().copy()

    # X = Raw Elo Difference (Home - Away)
    df_reg['elo_diff'] = df_reg['home_elo'] - df_reg['away_elo']
    X = df_reg[['elo_diff']]

    # y = Actual Margin of Victory (Points)
    df_reg['MOV'] = df_reg['home_score'] - df_reg['away_score']
    y = df_reg['MOV']

    # 3. Train Linear Regression
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    # 4. Extract Constants
    hfa_points = model.intercept_  # This is 'b' (Value of Home Field when Elos are equal)
    slope = model.coef_[0]         # This is 'm' (How many points 1 Elo unit is worth)

    elo_divisor = 1 / slope
    hfa_elo_adjustment = hfa_points * elo_divisor

    return hfa_elo_adjustment, elo_divisor

def calculate_elo_history(df, k=35, HFA_elo_adjustment=0, initial_rating=1500):
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
            shift = elo_change_calc(winner_elo=home_rating+HFA_elo_adjustment, loser_elo=away_rating, k=k)
            current_elos[home] += shift
            current_elos[away] -= shift
            
        elif row['away_score'] > row['home_score']: # away win
            shift = elo_change_calc(winner_elo=away_rating, loser_elo=home_rating+HFA_elo_adjustment, k=k)
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
from sklearn.base import BaseEstimator, RegressorMixin

class EloEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, k=20, initial_rating=1500):
        self.k = k
        self.initial_rating = initial_rating
        self.team_ratings_ = {}

    def fit(self, X, y=None):
        # X is expected to be the elo_prep DataFrame
        all_teams = set(X['home_team']).union(set(X['away_team']))
        self.team_ratings_ = {team: self.initial_rating for team in all_teams}
        
        for index, row in X.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Ensure teams exist
            if home not in self.team_ratings_: self.team_ratings_[home] = self.initial_rating
            if away not in self.team_ratings_: self.team_ratings_[away] = self.initial_rating
            
            home_rating = self.team_ratings_[home]
            away_rating = self.team_ratings_[away]
            
            if row['home_score'] > row['away_score']:
                shift = elo_change_calc(home_rating, away_rating, self.k)
                self.team_ratings_[home] += shift
                self.team_ratings_[away] -= shift
            elif row['away_score'] > row['home_score']:
                shift = elo_change_calc(away_rating, home_rating, self.k)
                self.team_ratings_[away] += shift
                self.team_ratings_[home] -= shift
        
        return self

    def score(self, X, y=None):
        brier_scores = []
        current_ratings = self.team_ratings_.copy()
        
        for index, row in X.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            if home not in current_ratings: current_ratings[home] = self.initial_rating
            if away not in current_ratings: current_ratings[away] = self.initial_rating
            
            home_rating = current_ratings[home]
            away_rating = current_ratings[away]
            
            prob_home_win = 1 / (1 + 10**((away_rating - home_rating) / 400))
            home_win = 1 if row['home_score'] > row['away_score'] else 0
            
            
            brier_scores.append((prob_home_win - home_win) ** 2)
            
            # Update ratings for next prediction (sequential evaluation)
            if row['home_score'] > row['away_score']:
                shift = elo_change_calc(home_rating, away_rating, self.k)
                current_ratings[home] += shift
                current_ratings[away] -= shift
            elif row['away_score'] > row['home_score']:
                shift = elo_change_calc(away_rating, home_rating, self.k)
                current_ratings[away] += shift
                current_ratings[home] -= shift
                
        return -np.mean(brier_scores) # Higher is better in sklearn, so negative Brier score

