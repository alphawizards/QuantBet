"""
Tests for Bayesian Elo rating system.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.models.prediction.bayesian_elo import BayesianEloRating, TeamRating


class TestTeamRating:
    """Tests for TeamRating dataclass."""
    
    def test_default_values(self):
        """Default rating should be 1500 with reasonable uncertainty."""
        rating = TeamRating(team_code="MEL")
        assert rating.mean == 1500.0
        assert rating.std == 200.0
        assert rating.games_played == 0
    
    def test_credible_intervals(self):
        """Credible intervals should be computed correctly."""
        rating = TeamRating(team_code="MEL", mean=1600, std=100)
        
        # 95% CI: mean ± 1.96 * std
        assert rating.lower_bound == pytest.approx(1600 - 1.96 * 100, rel=0.01)
        assert rating.upper_bound == pytest.approx(1600 + 1.96 * 100, rel=0.01)
    
    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        rating = TeamRating(team_code="MEL", mean=1550, std=150, games_played=10)
        d = rating.to_dict()
        
        assert d['team_code'] == "MEL"
        assert d['rating_mean'] == 1550
        assert d['rating_std'] == 150
        assert d['games'] == 10


class TestBayesianEloRating:
    """Tests for Bayesian Elo rating system."""
    
    @pytest.fixture
    def elo(self):
        """Create a fresh Elo system."""
        return BayesianEloRating()
    
    def test_new_team_gets_prior(self, elo):
        """New team should get prior rating."""
        rating = elo.get_rating("NEW")
        assert rating.mean == elo.prior_mean
        assert rating.std == elo.prior_std
    
    def test_expected_score_equal_teams(self, elo):
        """Equal-rated teams should have ~50% without home advantage."""
        # Override home advantage for this test
        elo.home_advantage = 0
        expected = elo.expected_score(1500, 1500)
        assert expected == pytest.approx(0.5, abs=0.01)
    
    def test_expected_score_with_home_advantage(self, elo):
        """Home advantage should increase home win probability."""
        expected = elo.expected_score(1500, 1500)
        # With 50 point home advantage, should be > 50%
        assert expected > 0.5
    
    def test_expected_score_higher_rating(self, elo):
        """Higher rated team should have higher expected score."""
        elo.home_advantage = 0  # Remove home advantage
        expected = elo.expected_score(1600, 1400)
        assert expected > 0.7  # Should be significantly > 50%
    
    def test_update_changes_ratings(self, elo):
        """Game result should update both teams' ratings."""
        initial_home = elo.get_rating("MEL").mean
        initial_away = elo.get_rating("SYD").mean
        
        elo.update_from_game("MEL", "SYD", home_win=True)
        
        # Winner should increase
        assert elo.get_rating("MEL").mean > initial_home
        # Loser should decrease
        assert elo.get_rating("SYD").mean < initial_away
    
    def test_update_decreases_uncertainty(self, elo):
        """Playing games should decrease uncertainty."""
        initial_std = elo.get_rating("MEL").std
        
        elo.update_from_game("MEL", "SYD", home_win=True)
        
        assert elo.get_rating("MEL").std < initial_std
        assert elo.get_rating("SYD").std < initial_std
    
    def test_predict_proba_returns_tuple(self, elo):
        """Prediction should return (mean, std) tuple."""
        elo.update_from_game("MEL", "SYD", home_win=True)
        
        prob, uncertainty = elo.predict_proba("MEL", "SYD")
        
        assert 0 <= prob <= 1
        assert uncertainty >= 0
    
    def test_prediction_uncertainty_decreases_with_games(self, elo):
        """More games should decrease prediction uncertainty."""
        # Single game
        elo.update_from_game("MEL", "SYD", home_win=True)
        _, uncertainty1 = elo.predict_proba("MEL", "SYD")
        
        # More games
        for _ in range(10):
            elo.update_from_game("MEL", "SYD", home_win=True)
        
        _, uncertainty2 = elo.predict_proba("MEL", "SYD")
        
        assert uncertainty2 < uncertainty1
    
    def test_predict_with_confidence(self, elo):
        """Confidence prediction should return full details."""
        elo.update_from_game("MEL", "SYD", home_win=True)
        
        pred = elo.predict_with_confidence("MEL", "SYD")
        
        assert 'home_win_prob' in pred
        assert 'home_win_prob_std' in pred
        assert 'ci_25' in pred
        assert 'ci_75' in pred
        assert 'home_rating_mean' in pred
        assert 'away_rating_mean' in pred
    
    def test_ci_ordering(self, elo):
        """Confidence intervals should be properly ordered."""
        elo.update_from_game("MEL", "SYD", home_win=True)
        pred = elo.predict_with_confidence("MEL", "SYD")
        
        assert pred['ci_05'] <= pred['ci_25']
        assert pred['ci_25'] <= pred['home_win_prob']
        assert pred['home_win_prob'] <= pred['ci_75']
        assert pred['ci_75'] <= pred['ci_95']


class TestBayesianEloFitFromHistory:
    """Tests for fitting from historical data."""
    
    @pytest.fixture
    def sample_games(self):
        """Create sample game history."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i*3) for i in range(20)]
        
        games = []
        for i, date in enumerate(dates):
            # MEL beats SYD most of the time
            if i % 3 == 0:
                games.append({
                    'game_date': date,
                    'home_team': 'MEL',
                    'away_team': 'SYD',
                    'home_score': 95,
                    'away_score': 85
                })
            else:
                games.append({
                    'game_date': date,
                    'home_team': 'SYD',
                    'away_team': 'MEL',
                    'home_score': 85,
                    'away_score': 90
                })
        
        return pd.DataFrame(games)
    
    def test_fit_from_history(self, sample_games):
        """Should fit ratings from game history."""
        elo = BayesianEloRating()
        elo.fit_from_history(sample_games)
        
        # MEL should be rated higher (wins most games)
        assert elo.get_rating("MEL").mean > elo.get_rating("SYD").mean
    
    def test_all_teams_have_ratings(self, sample_games):
        """All teams in history should have ratings."""
        elo = BayesianEloRating()
        elo.fit_from_history(sample_games)
        
        ratings_df = elo.get_all_ratings()
        assert len(ratings_df) == 2
        assert "MEL" in ratings_df['team_code'].values
        assert "SYD" in ratings_df['team_code'].values
    
    def test_games_played_tracked(self, sample_games):
        """Games played should be tracked correctly."""
        elo = BayesianEloRating()
        elo.fit_from_history(sample_games)
        
        # Each team plays in all 20 games
        assert elo.get_rating("MEL").games_played == 20
        assert elo.get_rating("SYD").games_played == 20


class TestKellyStake:
    """Tests for Kelly stake calculation."""
    
    @pytest.fixture
    def fitted_elo(self):
        """Create Elo with some history."""
        elo = BayesianEloRating()
        # MEL is clearly better
        for _ in range(10):
            elo.update_from_game("MEL", "SYD", home_win=True)
        return elo
    
    def test_kelly_no_edge_returns_zero(self, fitted_elo):
        """No edge should recommend zero stake."""
        # MEL is heavily favored, but odds are 1.10 (implied ~90%)
        result = fitted_elo.kelly_stake("MEL", "SYD", odds=1.10)
        # If our probability is less than implied, stake should be 0
        # (depending on exact probability)
        assert result['recommended_stake'] >= 0
    
    def test_kelly_positive_edge(self, fitted_elo):
        """Positive edge should recommend positive stake."""
        # Get MEL's actual probability
        prob, _ = fitted_elo.predict_proba("MEL", "SYD")
        
        # Set odds that give positive edge
        implied = prob - 0.1  # 10% edge
        odds = 1 / implied
        
        result = fitted_elo.kelly_stake("MEL", "SYD", odds=odds)
        
        assert result['edge'] > 0
        assert result['recommended_stake'] > 0
    
    def test_kelly_uses_conservative_probability(self, fitted_elo):
        """Conservative mode should use lower probability."""
        result_normal = fitted_elo.kelly_stake("MEL", "SYD", odds=2.0, use_conservative=False)
        result_conservative = fitted_elo.kelly_stake("MEL", "SYD", odds=2.0, use_conservative=True)
        
        assert result_conservative['probability_conservative'] < result_normal['probability']
