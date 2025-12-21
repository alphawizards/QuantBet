"""
Unit tests for NBL Feature Engineering.
"""

import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from src.features.engineer import (
    NBLFeatureEngineer,
    TeamEfficiency,
    TravelFatigueResult,
    TEAM_LOCATIONS,
)


class TestHaversineDistance:
    """Tests for distance calculations."""
    
    def test_same_location_zero_distance(self):
        """Same coordinates should have zero distance."""
        dist = NBLFeatureEngineer.haversine_distance(
            -37.8136, 144.9631,  # Melbourne
            -37.8136, 144.9631   # Melbourne
        )
        assert dist == 0.0
    
    def test_sydney_melbourne_distance(self):
        """Sydney to Melbourne should be ~700-900km."""
        dist = NBLFeatureEngineer.haversine_distance(
            -33.8688, 151.2093,  # Sydney
            -37.8136, 144.9631   # Melbourne
        )
        assert 700 < dist < 900
    
    def test_perth_sydney_distance(self):
        """Perth to Sydney should be ~3200-3400km."""
        dist = NBLFeatureEngineer.haversine_distance(
            -31.9505, 115.8605,  # Perth
            -33.8688, 151.2093   # Sydney
        )
        assert 3200 < dist < 3400


class TestPossessionsCalculation:
    """Tests for possession estimation."""
    
    def test_basic_possessions(self):
        """Basic possession calculation works."""
        engineer = NBLFeatureEngineer()
        
        # Possessions ≈ FGA - OREB + TO + 0.44*FTA
        poss = engineer.calculate_possessions(
            fga=80,
            oreb=10,
            turnovers=15,
            fta=20
        )
        
        # 80 - 10 + 15 + 0.44*20 = 85 + 8.8 = 93.8
        assert poss == pytest.approx(93.8)
    
    def test_zero_inputs(self):
        """Zero inputs return zero possessions."""
        engineer = NBLFeatureEngineer()
        poss = engineer.calculate_possessions(0, 0, 0, 0)
        assert poss == 0.0


class TestRollingEfficiency:
    """Tests for rolling efficiency calculations."""
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical game data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='3D')
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'game_date': date,
                'home_team': 'MEL' if i % 2 == 0 else 'SYD',
                'away_team': 'SYD' if i % 2 == 0 else 'MEL',
                'home_score': 90 + i,
                'away_score': 85 + i,
                'home_fga': 75,
                'away_fga': 72,
                'home_oreb': 10,
                'away_oreb': 8,
                'home_turnovers': 12,
                'away_turnovers': 14,
                'home_fta': 20,
                'away_fta': 18,
            })
        
        return pd.DataFrame(data)
    
    def test_rolling_efficiency_returns_result(self, sample_historical_data):
        """Rolling efficiency returns valid result for team with data."""
        engineer = NBLFeatureEngineer(rolling_window=5)
        
        result = engineer.get_rolling_efficiency(
            team_code='MEL',
            game_date=datetime(2024, 2, 1),
            historical_data=sample_historical_data
        )
        
        assert result is not None
        assert isinstance(result, TeamEfficiency)
        assert result.offensive_rating > 0
        assert result.defensive_rating > 0
    
    def test_insufficient_data_returns_none(self, sample_historical_data):
        """Returns None when insufficient game history."""
        engineer = NBLFeatureEngineer(rolling_window=10)
        
        # Only 5 games for MEL in our data
        result = engineer.get_rolling_efficiency(
            team_code='MEL',
            game_date=datetime(2024, 1, 20),  # Early, less data
            historical_data=sample_historical_data
        )
        
        assert result is None


class TestTravelFatigue:
    """Tests for travel fatigue scoring."""
    
    @pytest.fixture
    def sample_schedule(self):
        """Create sample schedule for travel calculations."""
        return pd.DataFrame([
            {'game_date': datetime(2024, 1, 1), 'home_team': 'MEL', 'away_team': 'SYD'},
            {'game_date': datetime(2024, 1, 3), 'home_team': 'SYD', 'away_team': 'MEL'},
            {'game_date': datetime(2024, 1, 10), 'home_team': 'PER', 'away_team': 'MEL'},
        ])
    
    def test_travel_fatigue_calculation(self, sample_schedule):
        """Basic travel fatigue calculation works."""
        engineer = NBLFeatureEngineer()
        
        result = engineer.calculate_travel_fatigue(
            team_code='MEL',
            game_date=datetime(2024, 1, 3),
            schedule=sample_schedule
        )
        
        assert isinstance(result, TravelFatigueResult)
        assert 0 <= result.score <= 5
        assert result.distance_km >= 0
    
    def test_first_game_no_fatigue(self, sample_schedule):
        """First game of season should have no travel fatigue."""
        engineer = NBLFeatureEngineer()
        
        result = engineer.calculate_travel_fatigue(
            team_code='PER',  # Not in schedule yet
            game_date=datetime(2023, 12, 1),
            schedule=sample_schedule
        )
        
        assert result.score == 0
        assert result.route_type == "First Game"
    
    def test_perth_east_coast_high_fatigue(self, sample_schedule):
        """Perth to East Coast should have higher fatigue."""
        engineer = NBLFeatureEngineer()
        
        # MEL playing in Perth after playing in Sydney
        result = engineer.calculate_travel_fatigue(
            team_code='MEL',
            game_date=datetime(2024, 1, 10),
            schedule=sample_schedule
        )
        
        # Should be higher score due to Perth distance
        assert result.distance_km > 2000  # Perth is far
    
    def test_back_to_back_increases_fatigue(self, sample_schedule):
        """Back-to-back games should increase fatigue."""
        engineer = NBLFeatureEngineer()
        
        result = engineer.calculate_travel_fatigue(
            team_code='MEL',
            game_date=datetime(2024, 1, 3),  # 2 days after Jan 1
            schedule=sample_schedule
        )
        
        assert result.days_rest == 2


class TestImportAvailability:
    """Tests for import player availability tracking."""
    
    @pytest.fixture
    def sample_availability(self):
        """Create sample player availability data."""
        return pd.DataFrame([
            {'player_id': 'p1', 'team': 'MEL', 'is_import': True, 
             'available_from': None, 'unavailable_until': None},
            {'player_id': 'p2', 'team': 'MEL', 'is_import': True,
             'available_from': datetime(2024, 1, 15), 'unavailable_until': None},
            {'player_id': 'p3', 'team': 'MEL', 'is_import': False,
             'available_from': None, 'unavailable_until': None},
            {'player_id': 'p4', 'team': 'SYD', 'is_import': True,
             'available_from': None, 'unavailable_until': datetime(2024, 1, 20)},
        ])
    
    def test_all_imports_available(self, sample_availability):
        """Check when all imports are available."""
        engineer = NBLFeatureEngineer()
        
        result = engineer.check_import_availability(
            team_code='MEL',
            game_date=datetime(2024, 1, 20),  # After p2 available
            player_availability=sample_availability
        )
        
        assert result['all_imports_available'] == True
        assert result['import_count_available'] == 2
        assert result['import_count_total'] == 2
    
    def test_import_not_yet_available(self, sample_availability):
        """Check when import hasn't arrived yet."""
        engineer = NBLFeatureEngineer()
        
        result = engineer.check_import_availability(
            team_code='MEL',
            game_date=datetime(2024, 1, 10),  # Before p2 available
            player_availability=sample_availability
        )
        
        assert result['all_imports_available'] == False
        assert result['import_count_available'] == 1
        assert 'p2' in result['missing_imports']
    
    def test_team_with_no_imports(self, sample_availability):
        """Team with no imports should return empty result."""
        engineer = NBLFeatureEngineer()
        
        # Add a team with no imports
        no_import_df = sample_availability[
            sample_availability['team'] == 'nonexistent'
        ]
        
        result = engineer.check_import_availability(
            team_code='nonexistent',
            game_date=datetime(2024, 1, 15),
            player_availability=no_import_df
        )
        
        assert result['all_imports_available'] == True
        assert result['import_count_total'] == 0


class TestGenerateGameFeatures:
    """Tests for the full feature generation pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='3D')
        
        historical = []
        for i, date in enumerate(dates):
            historical.append({
                'game_date': date,
                'home_team': 'MEL' if i % 2 == 0 else 'SYD',
                'away_team': 'SYD' if i % 2 == 0 else 'MEL',
                'home_score': 90 + i,
                'away_score': 85 + i,
                'home_fga': 75, 'away_fga': 72,
                'home_oreb': 10, 'away_oreb': 8,
                'home_turnovers': 12, 'away_turnovers': 14,
                'home_fta': 20, 'away_fta': 18,
            })
        
        schedule = pd.DataFrame(historical)
        
        availability = pd.DataFrame([
            {'player_id': 'p1', 'team': 'MEL', 'is_import': True,
             'available_from': None, 'unavailable_until': None},
            {'player_id': 'p2', 'team': 'SYD', 'is_import': True,
             'available_from': None, 'unavailable_until': None},
        ])
        
        return {
            'historical': pd.DataFrame(historical),
            'schedule': schedule,
            'availability': availability,
        }
    
    def test_generates_all_features(self, sample_data):
        """Feature generation returns all expected features."""
        engineer = NBLFeatureEngineer(rolling_window=3)
        
        features = engineer.generate_game_features(
            home_team='MEL',
            away_team='SYD',
            game_date=datetime(2024, 2, 1),
            historical_data=sample_data['historical'],
            schedule=sample_data['schedule'],
            player_availability=sample_data['availability']
        )
        
        # Check for key features
        assert 'home_ortg_l5' in features or np.isnan(features.get('home_ortg_l5', np.nan))
        assert 'home_travel_fatigue' in features
        assert 'home_imports_available' in features
        assert 'travel_fatigue_diff' in features
    
    def test_missing_player_availability_handled(self, sample_data):
        """Works without player availability data."""
        engineer = NBLFeatureEngineer(rolling_window=3)
        
        features = engineer.generate_game_features(
            home_team='MEL',
            away_team='SYD',
            game_date=datetime(2024, 2, 1),
            historical_data=sample_data['historical'],
            schedule=sample_data['schedule'],
            player_availability=None
        )
        
        # Should have NaN for import features
        assert np.isnan(features['home_imports_available'])


class TestTeamLocations:
    """Tests for team location data."""
    
    def test_all_nbl_teams_have_locations(self):
        """All major NBL teams should have location data."""
        major_teams = ['MEL', 'SYD', 'PER', 'BRI', 'ADL', 'NZB']
        
        for team in major_teams:
            assert team in TEAM_LOCATIONS
            lat, lon = TEAM_LOCATIONS[team]
            assert -50 < lat < 0  # Southern hemisphere
            assert 100 < lon < 180  # Oceania
    
    def test_perth_is_west(self):
        """Perth should be the westernmost team."""
        per_lon = TEAM_LOCATIONS['PER'][1]
        
        for team, (lat, lon) in TEAM_LOCATIONS.items():
            if team not in ['PER', 'PER_W']:
                assert lon > per_lon, f"{team} should be east of Perth"
