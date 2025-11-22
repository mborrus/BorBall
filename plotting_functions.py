import matplotlib.pyplot as plt
import numpy as np

def plot_elo_every_season(teams, chart):
    plt.figure(figsize=(16, 8)) # Wide chart to fit all the years
    
    # 1. Plot the Lines
    # We reset index to get a simple 0, 1, 2... x-axis for plotting
    plot_data = chart[teams].reset_index(drop=True)
    for team in teams:
        plt.plot(plot_data.index, plot_data[team], label=team, linewidth=2)
    
    # 2. Calculate Tick Locations
    # We look at the original MultiIndex to find where the season changes
    # The index is (Season, Week). We want the integer position of the first appearance of each season.
    
    # Get all unique seasons
    all_seasons = chart.index.get_level_values('season').unique()
    
    tick_locations = []
    tick_labels = []
    
    for season in all_seasons:
        # Find the integer location (iloc) of the first week of this season
        # We search for the first occurrence of the season in the index
        try:
            # Get the first index tuple for this season (e.g., (2004, 1))
            first_idx = chart.loc[season].index[0] 
            # Find the integer position of (season, first_idx) in the full chart
            # chart.index.get_loc returns a slice if there are duplicates, or int if unique
            # Since our index is unique (Season, Week), it should work, but let's be safe:
            # A safer way is to use the reset_index version we used for plotting:
            
            # Find the first row where 'season' == current_season
            season_start_loc = chart.index.get_level_values('season').tolist().index(season)
            
            tick_locations.append(season_start_loc)
            tick_labels.append(str(season))
        except:
            continue

    # 3. Apply the Ticks
    plt.xticks(tick_locations, tick_labels, rotation=90, fontsize=10)
    
    # 4. Styling
    plt.axhline(1500, color='black', linestyle='--', alpha=0.5, label="Average")
    plt.title(f"Elo History: {', '.join(teams)}", fontsize=16)
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
