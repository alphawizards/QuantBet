"""
Unit Tests for Feature Validator.

Tests all validation rules to ensure features are properly validated
before being passed to models.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from src.features.validator import (
    FeatureValidator,
    ValidationIssue,
    validate_features
)


@dataclass
class MockFeatures:
    """Mock features class for testing."""
    home_elo: float = 1625.0
    away_elo: float = 1550.0
    elo_diff: float = 75.0
    home_l5_win_pct: float = 0.8
    away_l5_win_pct: float = 0.6
    home_rest_days: int = 3
    away_rest_days: int = 1
    away_travel_km: float = 250.0


class TestFeatureValidator:
    """Test suite for FeatureValidator."""
    
    def test_valid_features_pass(self):
        """Test that valid features pass validation."""
        features = MockFeatures()
        validator = FeatureValidator()
        
        assert validator.validate(features) == True
        assert len(validator.issues) == 0
    
    def test_nan_detection(self):
        """Test that NaN values are detected."""
        features = MockFeatures(home_elo=np.nan)
        validator = FeatureValidator()
        
        with pytest.raises(ValueError, match="Feature validation failed"):
            validator.validate(features)
        
        assert len(validator.issues) == 1
        assert validator.issues[0].issue_type == 'nan'
        assert validator.issues[0].field_name == 'home_elo'
    
    def test_infinity_detection(self):
        """Test that infinity values are detected."""
        features = MockFeatures(elo_diff=np.inf)
        validator = FeatureValidator()
        
        with pytest.raises(ValueError, match="Feature validation failed"):
            validator.validate(features)
        
        assert len(validator.issues) == 1
        assert validator.issues[0].issue_type == 'infinity'
    
    def test_probability_range_violation(self):
        """Test that probability fields outside [0,1] are detected."""
        features = MockFeatures(home_l5_win_pct=1.5)
        validator = FeatureValidator()
        
        with pytest.raises(ValueError):
            validator.validate(features)
        
        assert any(issue.issue_type == 'range' for issue in validator.issues)
    
    def test_negative_value_detection(self):
        """Test that negative values in non-negative fields are detected."""
        features = MockFeatures(home_rest_days=-1)
        validator = FeatureValidator()
        
        with pytest.raises(ValueError):
            validator.validate(features)
        
        assert any(
            issue.field_name == 'home_rest_days' and issue.issue_type == 'range'
            for issue in validator.issues
        )
    
    def test_multiple_issues_detected(self):
        """Test that multiple issues are all detected."""
        features = MockFeatures(
            home_elo=np.nan,
            home_l5_win_pct=1.2,
            home_rest_days=-5
        )
        validator = FeatureValidator()
        
        with pytest.raises(ValueError):
            validator.validate(features)
        
        assert len(validator.issues) == 3
    
    def test_non_strict_mode_allows_issues(self):
        """Test that non-strict mode logs issues but doesn't raise."""
        features = MockFeatures(home_elo=np.nan)
        validator = FeatureValidator(strict=False)
        
        # Should not raise exception
        result = validator.validate(features)
        
        assert result == False  # Validation failed
        assert len(validator.issues) == 1  # But issue was logged
    
    def test_required_fields_checked(self):
        """Test that required fields are checked."""
        @dataclass
        class IncompleteFeatures:
            home_elo: float = 1625.0
            # missing away_elo
        
        features = IncompleteFeatures()
        validator = FeatureValidator()
        
        with pytest.raises(ValueError):
            validator.validate(features)
        
        assert any(
            issue.issue_type == 'missing' and issue.field_name == 'away_elo'
            for issue in validator.issues
        )
    
    def test_convenience_function(self):
        """Test the validate_features convenience function."""
        features = MockFeatures()
        
        # Should work with valid features
        assert validate_features(features) == True
        
        # Should raise with invalid features
        invalid_features = MockFeatures(home_elo=np.nan)
        with pytest.raises(ValueError):
            validate_features(invalid_features)
    
    def test_issue_formatting(self):
        """Test that issues are formatted correctly."""
        features = MockFeatures(home_elo=np.nan, home_l5_win_pct=1.5)
        validator = FeatureValidator()
        
        try:
            validator.validate(features)
        except ValueError as e:
            error_msg = str(e)
            assert "home_elo" in error_msg
            assert "nan" in error_msg
            assert "home_l5_win_pct" in error_msg
            assert "range" in error_msg


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_values_allowed(self):
        """Test that legitimate zero values are allowed."""
        features = MockFeatures(home_rest_days=0, elo_diff=0.0)
        validator = FeatureValidator()
        
        assert validator.validate(features) == True
    
    def test_boundary_probabilities(self):
        """Test that 0.0 and 1.0 probabilities are valid."""
        features = MockFeatures(home_l5_win_pct=0.0)
        validator = FeatureValidator()
        assert validator.validate(features) == True
        
        features = MockFeatures(away_l5_win_pct=1.0)
        validator = FeatureValidator()
        assert validator.validate(features) == True
    
    def test_very_large_but_finite_values(self):
        """Test that very large (but finite) values are allowed."""
        features = MockFeatures(away_travel_km=10000.0)
        validator = FeatureValidator()
        
        assert validator.validate(features) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
