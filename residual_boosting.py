
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def prepare_residual_data(elo_df, games_df):
    """
    Prepares data for residual boosting.
    1. Merges Elo predictions with game features.
    2. Calculates residuals (Actual Margin - Predicted Margin).
    3. Prepares features (stadium, week, wind, temp).
    """
    # Merge to get features
    # elo_df should have 'game_id', 'home_elo', 'away_elo'
    # games_df should have 'game_id', 'stadium_id', 'week', 'temp', 'wind', 'roof'
    
    # Identify columns to bring from games_df
    # We don't want to duplicate columns that are already in elo_df (like 'week')
    potential_cols = ['game_id', 'stadium_id', 'week', 'temp', 'wind', 'roof']
    cols_to_use = ['game_id'] # Always need join key
    
    for c in potential_cols:
        if c == 'game_id': continue
        if c in games_df.columns and c not in elo_df.columns:
            cols_to_use.append(c)
        elif c in games_df.columns and c in elo_df.columns:
            # If it's in both, we assume elo_df has the correct one (e.g. week)
            # So we don't add it to cols_to_use
            pass
            
    merged_df = pd.merge(elo_df, games_df[cols_to_use], on='game_id', how='left')
    
    # Calculate Predicted Margin based on Elo Difference
    # Using standard approximation: Margin = (HomeElo - AwayElo) / 25
    merged_df['elo_diff'] = merged_df['home_elo'] - merged_df['away_elo']
    merged_df['predicted_margin'] = merged_df['elo_diff'] / 25.0
    
    # Actual Margin
    merged_df['actual_margin'] = merged_df['home_score'] - merged_df['away_score']
    
    # Residual
    merged_df['residual'] = merged_df['actual_margin'] - merged_df['predicted_margin']
    
    # Handle Missing Values
    if 'temp' in merged_df.columns:
        merged_df['temp'] = pd.to_numeric(merged_df['temp'], errors='coerce')
        merged_df['temp'] = merged_df['temp'].fillna(70)
        
    if 'wind' in merged_df.columns:
        merged_df['wind'] = pd.to_numeric(merged_df['wind'], errors='coerce')
        merged_df['wind'] = merged_df['wind'].fillna(0)
    
    # Encode Categoricals
    if 'stadium_id' in merged_df.columns:
        le = LabelEncoder()
        # Handle missing stadium_ids
        merged_df['stadium_id'] = merged_df['stadium_id'].fillna('UNKNOWN')
        merged_df['stadium_id_encoded'] = le.fit_transform(merged_df['stadium_id'].astype(str))
    
    return merged_df

def train_residual_models(df):
    """
    Trains Random Forest and Boosting models to predict residuals.
    """
    # Define potential features
    potential_features = ['week', 'temp', 'wind', 'stadium_id_encoded']
    target = 'residual'
    
    # Select only features that exist in the dataframe
    features = [f for f in potential_features if f in df.columns]
    
    if not features:
        print("No features available for residual boosting.")
        return [], []
        
    print(f"Using features: {features}")
    
    # Filter for rows with all features
    df_model = df.dropna(subset=features + [target]).copy()
    
    if len(df_model) == 0:
        print("No data remaining after dropping NaNs.")
        return [], []
    
    X = df_model[features]
    y = df_model[target]
    
    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_scores = []
    gb_scores = []
    xgb_scores = []
    
    print(f"Training models on {len(df_model)} games...")
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_scores.append(mean_absolute_error(y_test, y_pred_rf))
        
        # Gradient Boosting (sklearn)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        gb_scores.append(mean_absolute_error(y_test, y_pred_gb))
        
        # XGBoost (if available)
        if HAS_XGB:
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            xgb_scores.append(mean_absolute_error(y_test, y_pred_xgb))
            
    print(f"Random Forest MAE: {np.mean(rf_scores):.4f}")
    print(f"Gradient Boosting MAE: {np.mean(gb_scores):.4f}")
    if HAS_XGB:
        print(f"XGBoost MAE: {np.mean(xgb_scores):.4f}")
    
    baseline_mae = np.mean(np.abs(y))
    print(f"Baseline (Elo only) MAE of Residual: {baseline_mae:.4f}")
    
    return rf_scores, gb_scores
