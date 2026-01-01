"""
Feature Validation Framework.

Validates GameFeatures before model input to prevent NaN/infinity crashes.
Critical for production stability.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ValidationIssue:
    """Represents a feature validation issue."""
    field_name: str
    issue_type: str  # 'nan', 'infinity', 'range', 'missing', 'type'
    value: Any
    expected: str
    severity: str = 'error'  # 'error' or 'warning'


class FeatureValidator:
    """
    Validate GameFeatures before prediction.
    
    Prevents crashes from NaN, infinity, or out-of-range values.
    
    Example:
        >>> validator = FeatureValidator()
        >>> features = GameFeatures(home_elo=1625, away_elo=1550, ...)
        >>> validator.validate(features)  # Raises ValueError if invalid
        True
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise exception on any issue.
                   If False, log warnings but allow validation to pass.
        """
        self.strict = strict
        self.issues: List[ValidationIssue] = []
    
    def validate(self, features: Any) -> bool:
        """
        Validate features object.
        
        Args:
            features: GameFeatures or similar object with feature attributes
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If strict=True and validation fails
        """
        self.issues = []
        
        # Check for NaN/infinity in numeric fields
        self._check_numeric_validity(features)
        
        # Check probability ranges [0, 1]
        self._check_probability_ranges(features)
        
        # Check non-negative fields
        self._check_non_negative(features)
        
        # Check required fields exist
        self._check_required_fields(features)
        
        # If strict mode and issues found, raise exception
        if self.strict and self.issues:
            raise ValueError(
                f"Feature validation failed with {len(self.issues)} issue(s):\n" +
                self._format_issues()
            )
        
        return len(self.issues) == 0
    
    def _check_numeric_validity(self, features: Any) -> None:
        """Check for NaN and infinity in floating point fields."""
        for name, value in features.__dict__.items():
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    self.issues.append(ValidationIssue(
                        field_name=name,
                        issue_type='nan',
                        value=value,
                        expected='finite number',
                        severity='error'
                    ))
                elif np.isinf(value):
                    self.issues.append(ValidationIssue(
                        field_name=name,
                        issue_type='infinity',
                        value=value,
                        expected='finite number',
                        severity='error'
                    ))
    
    def _check_probability_ranges(self, features: Any) -> None:
        """Check that probability fields are in [0, 1] range."""
        prob_fields = [
            'home_l5_win_pct', 'away_l5_win_pct',
            'home_l10_win_pct', 'away_l10_win_pct',
            'home_injury_impact', 'away_injury_impact'
        ]
        
        for field in prob_fields:
            if hasattr(features, field):
                val = getattr(features, field)
                if isinstance(val, (float, int)):
                    if not (0 <= val <= 1):
                        self.issues.append(ValidationIssue(
                            field_name=field,
                            issue_type='range',
                            value=val,
                            expected='0.0 to 1.0',
                            severity='error'
                        ))
    
    def _check_non_negative(self, features: Any) -> None:
        """Check that certain fields are non-negative."""
        non_neg_fields = [
            'home_rest_days', 'away_rest_days',
            'away_travel_km', 'game_number_in_season',
            'days_into_season'
        ]
        
        for field in non_neg_fields:
            if hasattr(features, field):
                val = getattr(features, field)
                if isinstance(val, (int, float)):
                    if val < 0:
                        self.issues.append(ValidationIssue(
                            field_name=field,
                            issue_type='range',
                            value=val,
                            expected='>= 0',
                            severity='error'
                        ))
    
    def _check_required_fields(self, features: Any) -> None:
        """Check that required fields are present."""
        required_fields = ['home_elo', 'away_elo']  # Minimum required
        
        for field in required_fields:
            if not hasattr(features, field):
                self.issues.append(ValidationIssue(
                    field_name=field,
                    issue_type='missing',
                    value=None,
                    expected='present',
                    severity='error'
                ))
    
    def _format_issues(self) -> str:
        """Format issues for error message."""
        lines = []
        for i, issue in enumerate(self.issues, 1):
            lines.append(
                f"  {i}. {issue.field_name}: {issue.issue_type} "
                f"(got {issue.value}, expected {issue.expected})"
            )
        return "\n".join(lines)
    
    def get_issues(self) -> List[ValidationIssue]:
        """Get list of validation issues found."""
        return self.issues.copy()


# Convenience function
def validate_features(features: Any, strict: bool = True) -> bool:
    """
    Validate features using default validator.
    
    Args:
        features: Features object to validate
        strict: If True, raise exception on validation failure
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If strict=True and validation fails
    """
    validator = FeatureValidator(strict=strict)
    return validator.validate(features)
