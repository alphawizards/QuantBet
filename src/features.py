import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Any

class NBLFeatureEngineer:
    """
    Feature Engineering class for NBL/WNBL betting system.
    Focuses on Rolling Efficiency, Travel Fatigue, and Import Availability.
    """

    def __init__(self, games_df: pd.DataFrame, player_stats_df: pd.DataFrame, team_metadata: Dict[int, Dict]):
        """
        Initialize with game logs and player stats.

        Args:
            games_df: DataFrame containing game metadata (game_id, date, home_team_id, away_team_id, venue).
            player_stats_df: DataFrame containing player box scores.
            team_metadata: Dictionary mapping team_id to metadata (e.g., home_venue_location).
        """
        self.games_df = games_df.sort_values('date')
        self.player_stats_df = player_stats_df
        self.team_metadata = team_metadata

    def calculate_rolling_efficiency(self, team_id: int, window: int = 5) -> pd.DataFrame:
        """
        Calculates Last N games Offensive/Defensive Rating.

        Offensive Rating = (Points Scored / Possessions) * 100
        Defensive Rating = (Points Allowed / Possessions) * 100

        Note: Possessions in FIBA/NBL are often estimated if not tracked directly.
        Basic Estimation: 0.96 * (FGA + Turnovers + 0.44 * FTA - Offensive Rebounds)
        For this method, we assume pre-calculated possessions or a simplified estimate exists in logs.
        """
        team_games = self.games_df[(self.games_df['home_team_id'] == team_id) | (self.games_df['away_team_id'] == team_id)].copy()

        # Calculate/Extract raw stats per game
        # Assuming games_df has score columns 'home_score', 'away_score'
        # And we need to join with aggregated player stats or assuming games_df has possession data
        # For this implementation, we will simulate the calculation columns if they don't exist

        results = []

        for idx, row in team_games.iterrows():
            is_home = row['home_team_id'] == team_id

            points_scored = row['home_score'] if is_home else row['away_score']
            points_allowed = row['away_score'] if is_home else row['home_score']

            # TODO: Insert API Endpoint here or use real possession data
            # Placeholder: estimating 75 possessions per game for NBL 40-min game
            possessions = 75.0

            off_rtg = (points_scored / possessions) * 100
            def_rtg = (points_allowed / possessions) * 100

            results.append({
                'game_id': row['game_id'],
                'date': row['date'],
                'team_id': team_id,
                'off_rtg': off_rtg,
                'def_rtg': def_rtg
            })

        stats_df = pd.DataFrame(results).sort_values('date')

        # Rolling averages
        stats_df['rolling_off_rtg'] = stats_df['off_rtg'].rolling(window=window, closed='left').mean()
        stats_df['rolling_def_rtg'] = stats_df['def_rtg'].rolling(window=window, closed='left').mean()

        return stats_df

    def calculate_travel_fatigue(self, team_id: int, current_game_date: pd.Timestamp, prev_game_venue: str, current_venue: str) -> int:
        """
        Calculates Travel_Fatigue_Score.

        Logic:
        - Base score: 0
        - Distance penalty: Function of distance between venues.
        - Rest bonus: Negative score for days of rest.
        - Specific weighting: Perth (PER) to New Zealand (NZL) leg is severe.
        """

        # Simplified distance mapping (in "units" of fatigue)
        # TODO: Replace with real geodistance calculation
        dist_matrix = {
            ('PER', 'NZL'): 10, # High fatigue
            ('NZL', 'PER'): 10,
            ('MEL', 'PER'): 4,
            ('SYD', 'PER'): 5,
            # ... other pairings
        }

        # Determine locations from venues (assuming venue strings contain city codes or are mapped)
        # For this snippet, we assume inputs are City Codes like 'PER', 'NZL'

        route = (prev_game_venue, current_venue)
        fatigue_score = dist_matrix.get(route, 1) # Default to 1 unit for local/short travel

        if prev_game_venue == current_venue:
            fatigue_score = 0

        # Perth-NZ specific check
        if set(route) == {'PER', 'NZL'}:
            fatigue_score *= 1.5 # 50% penalty multiplier for this specific harsh leg

        # Days rest
        # We need the previous game date from the dataframe
        team_games = self.games_df[(self.games_df['home_team_id'] == team_id) | (self.games_df['away_team_id'] == team_id)].sort_values('date')
        prev_games = team_games[team_games['date'] < current_game_date]

        if not prev_games.empty:
            last_game_date = prev_games.iloc[-1]['date']
            days_rest = (current_game_date - last_game_date).days
        else:
            days_rest = 7 # Assume well rested for first game

        # Formula: Fatigue - Days_Rest
        # More rest reduces fatigue.
        final_score = max(0, fatigue_score - (days_rest - 1)) # 1 day rest is standard, doesn't reduce much

        return int(final_score)

    def assess_import_availability(self, game_id: int, team_id: int) -> float:
        """
        Generates Import_Availability score.

        Checks if key 'Import' players (Americans/internationals) are active in player_stats.
        Returns a weighted score (0.0 to 1.0) representing % of available import impact.
        """

        # Get all players for the team
        # In a real scenario, we'd query a players metadata table to know who is an import.
        # Here we assume self.player_stats_df has an 'is_import' flag merged in or available.

        # For this exercise, let's assume we filter player_stats_df for the specific game and team
        # and checking a hypothetical 'is_import' column.

        # Mocking the data structure availability
        if 'is_import' not in self.player_stats_df.columns:
            # Fallback or error handling
            return 1.0 # Assume full strength if data missing

        game_stats = self.player_stats_df[
            (self.player_stats_df['game_id'] == game_id) &
            (self.player_stats_df['team_id'] == team_id)
        ]

        imports = game_stats[game_stats['is_import'] == True]

        if imports.empty:
            return 1.0 # No imports on roster?

        # Check active status
        # Using 'dnp' flag. dnp=False means they played.
        active_imports = imports[imports['dnp'] == False]

        # Simple ratio: Active Imports / Total Imports
        # Could weight by Usage Rate if available
        availability_score = len(active_imports) / len(imports)

        return availability_score
