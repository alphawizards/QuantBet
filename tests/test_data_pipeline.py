
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.models.features.feature_store import FeatureStore
from src.collectors.integration import NBLDataIntegrator

# Mock data generators
def generate_mock_match_results(n_games=10):
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_games)]
    data = {
        'match_id': [f'game_{i}' for i in range(n_games)],
        'season': ['2023-2024'] * n_games,
        'game_date': dates,
        'home_team': (['MEL', 'SYD'] * ((n_games // 2) + 1))[:n_games],
        'home_team_name': (['Melbourne United', 'Sydney Kings'] * ((n_games // 2) + 1))[:n_games],
        'away_team': (['SYD', 'MEL'] * ((n_games // 2) + 1))[:n_games],
        'away_team_name': (['Sydney Kings', 'Melbourne United'] * ((n_games // 2) + 1))[:n_games],
        'home_score': np.random.randint(80, 100, n_games),
        'away_score': np.random.randint(80, 100, n_games),
        'home_score_string': [str(x) for x in np.random.randint(80, 100, n_games)],
        'away_score_string': [str(x) for x in np.random.randint(80, 100, n_games)],
        'round_number': [1] * n_games,
        'venue_name': ['John Cain Arena'] * n_games,
        'match_time': dates  # Needed for integrator
    }
    return pd.DataFrame(data)

def generate_mock_odds_data(n_games=10):
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_games)]
    data = {
        'Date': dates,
        'Home Team': (['Melbourne United', 'Sydney Kings'] * ((n_games // 2) + 1))[:n_games],
        'Away Team': (['Sydney Kings', 'Melbourne United'] * ((n_games // 2) + 1))[:n_games],
        'Home Score': np.random.randint(80, 100, n_games),
        'Away Score': np.random.randint(80, 100, n_games),
        'Home Odds': [1.90] * n_games,
        'Away Odds': [1.90] * n_games,
        'Home Odds Open': [1.85] * n_games,
        'Home Odds Close': [1.90] * n_games,
        'Away Odds Open': [1.95] * n_games,
        'Away Odds Close': [1.90] * n_games,
        'Season': ['2023-24'] * n_games
    }
    return pd.DataFrame(data)

class TestDataPipelineIntegration:

    def test_integrator_merges_scraped_and_odds(self):
        """Test that NBLDataIntegrator correctly merges scraped results with xlsx odds."""
        integrator = NBLDataIntegrator()

        scraped_df = generate_mock_match_results()
        xlsx_df = generate_mock_odds_data()

        merged = integrator.merge_data_sources(scraped_df, xlsx_df)

        # Verify merge structure
        assert len(merged) == 10
        assert 'home_odds' in merged.columns
        assert 'venue_name' in merged.columns  # From scraped
        assert 'home_odds_close' in merged.columns  # From xlsx

        # Verify data integrity
        assert merged.iloc[0]['season'] == '2023-2024'  # NBLR format preserved (if preferred)

    def test_feature_store_consumes_merged_data(self):
        """Test that FeatureStore can generate features from the merged dataset."""
        integrator = NBLDataIntegrator()
        scraped_df = generate_mock_match_results(20) # More games for history
        xlsx_df = generate_mock_odds_data(20)

        # Create a unified DataFrame similar to what would be stored
        historical_data = integrator.merge_data_sources(scraped_df, xlsx_df)

        # Initialize FeatureStore
        store = FeatureStore()

        # Define a target game (the next one)
        target_game = {
            'game_id': 'game_new',
            'game_date': datetime(2023, 2, 1),
            'home_team': 'MEL',
            'away_team': 'SYD',
            'home_odds': 1.90,
            'away_odds': 1.90
        }
        target_series = pd.Series(target_game)

        # Generate feature vector
        # Note: FeatureStore.compute_features expects 'game' and 'past_games'
        # We need to ensure historical_data has the columns FeatureStore expects.
        # FeatureStore uses columns like 'home_score', 'away_score', 'home_team', 'away_team', 'game_date'

        # Ensure date format is compatible (datetime)
        historical_data['game_date'] = pd.to_datetime(historical_data['game_date'])

        vector = store.compute_features(
            game=target_series,
            historical_data=historical_data
        )

        # Verify feature vector generation
        features = vector.features

        # Check for key features
        assert 'home_win_rate_l5' in features
        assert 'away_win_rate_l5' in features
        assert 'h2h_home_wins' in features
        assert 'market_vig' in features

        # Check values are normalized/sane
        assert 0 <= features['home_win_rate_l5'] <= 1.0
        assert features['market_vig'] > 1.0  # Should be > 1.0 due to vig

    @patch('src.collectors.scraper.NBLDataScraper._download_rds')
    def test_scraper_caching_flow(self, mock_download):
        """Test that the scraper tries to cache data."""
        from src.collectors.scraper import NBLDataScraper
        import tempfile
        import shutil
        import os
        from pathlib import Path

        # Create temp cache dir
        temp_dir = tempfile.mkdtemp()
        try:
            scraper = NBLDataScraper(cache_dir=temp_dir, use_cache=True)

            # Mock the RDS return
            mock_df = pd.DataFrame({'a': [1, 2, 3]})
            mock_download.return_value = mock_df

            # First fetch - should download
            df1 = scraper._fetch_data("results_wide")
            assert mock_download.call_count == 1

            # Check file was created
            files = list(Path(temp_dir).glob("*.parquet"))
            assert len(files) == 1

            # Second fetch - should use cache (no download)
            df2 = scraper._fetch_data("results_wide")
            assert mock_download.call_count == 1  # Count shouldn't increase

            pd.testing.assert_frame_equal(df1, df2)

        finally:
            shutil.rmtree(temp_dir)
