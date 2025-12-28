"""
Probability Calibration for NBL/WNBL Predictions.

This module provides calibration methods to transform raw model outputs
into well-calibrated probabilities suitable for Kelly Criterion calculations.

Calibration is critical because:
    1. XGBoost outputs are not always well-calibrated
    2. Kelly Criterion is extremely sensitive to probability estimates
    3. Overconfident probabilities lead to over-betting and ruin
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class BaseCalibrator(ABC):
    """Abstract base class for probability calibrators."""
    
    @abstractmethod
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        """Fit the calibrator to data."""
        pass
    
    @abstractmethod
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities."""
        pass
    
    def fit_calibrate(
        self, 
        y_prob: np.ndarray, 
        y_true: np.ndarray
    ) -> np.ndarray:
        """Fit and return calibrated probabilities."""
        self.fit(y_prob, y_true)
        return self.calibrate(y_prob)


class PlattCalibrator(BaseCalibrator):
    """
    Platt Scaling (Sigmoid) Calibration.
    
    Fits a logistic regression model to map raw probabilities to calibrated
    probabilities using a sigmoid transformation:
    
    .. math::
        P(y=1|f) = \\frac{1}{1 + \\exp(Af + B)}
    
    Where f is the raw probability and A, B are learned parameters.
    
    This is ideal when:
        - The model is well-calibrated but needs minor adjustments
        - There's limited calibration data
        - You want a smooth calibration function
    
    Reference:
        Platt, J. (1999). Probabilistic outputs for support vector machines
        and comparisons to regularized likelihood methods.
    
    Example:
        >>> calibrator = PlattCalibrator()
        >>> calibrator.fit(train_probs, train_labels)
        >>> calibrated = calibrator.calibrate(test_probs)
    """
    
    def __init__(self):
        """Initialize Platt calibrator."""
        self._lr: Optional[LogisticRegression] = None
        self._is_fitted = False
    
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        """
        Fit Platt scaling model.
        
        Args:
            y_prob: Raw probability predictions, shape (n_samples,)
            y_true: True binary labels, shape (n_samples,)
        
        Returns:
            Self for method chaining
        """
        # Reshape for sklearn
        X = y_prob.reshape(-1, 1)
        
        # Use logit transform for better numerical stability
        # Clip to avoid log(0) or log(1)
        X_clipped = np.clip(X, 1e-10, 1 - 1e-10)
        X_logit = np.log(X_clipped / (1 - X_clipped))
        
        self._lr = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            C=1e10  # Minimal regularization
        )
        self._lr.fit(X_logit, y_true)
        self._is_fitted = True
        
        return self
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to raw probabilities.
        
        Args:
            y_prob: Raw probability predictions, shape (n_samples,)
        
        Returns:
            Calibrated probabilities
        
        Raises:
            RuntimeError: If calibrator has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before calibrating")
        
        X = y_prob.reshape(-1, 1)
        X_clipped = np.clip(X, 1e-10, 1 - 1e-10)
        X_logit = np.log(X_clipped / (1 - X_clipped))
        
        return self._lr.predict_proba(X_logit)[:, 1]
    
    @property
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted."""
        return self._is_fitted


class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic Regression Calibration.
    
    Fits a non-parametric, monotonically increasing function to map
    raw probabilities to calibrated probabilities.
    
    This is ideal when:
        - The calibration function may be non-linear
        - There's sufficient calibration data (prevents overfitting)
        - You don't want to assume a particular functional form
    
    Caution:
        Isotonic regression can overfit with small datasets. Prefer
        Platt scaling when n < 1000.
    
    Reference:
        Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores
        into accurate multiclass probability estimates.
    
    Example:
        >>> calibrator = IsotonicCalibrator()
        >>> calibrator.fit(train_probs, train_labels)
        >>> calibrated = calibrator.calibrate(test_probs)
    """
    
    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Initialize Isotonic calibrator.
        
        Args:
            out_of_bounds: How to handle values outside training range
                          Options: 'clip', 'nan', 'raise'
        """
        self._ir: Optional[IsotonicRegression] = None
        self._out_of_bounds = out_of_bounds
        self._is_fitted = False
    
    def fit(
        self, 
        y_prob: np.ndarray, 
        y_true: np.ndarray
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression model.
        
        Args:
            y_prob: Raw probability predictions, shape (n_samples,)
            y_true: True binary labels, shape (n_samples,)
        
        Returns:
            Self for method chaining
        """
        self._ir = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds=self._out_of_bounds
        )
        self._ir.fit(y_prob, y_true)
        self._is_fitted = True
        
        return self
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression to raw probabilities.
        
        Args:
            y_prob: Raw probability predictions, shape (n_samples,)
        
        Returns:
            Calibrated probabilities
        
        Raises:
            RuntimeError: If calibrator has not been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before calibrating")
        
        return self._ir.predict(y_prob)
    
    @property
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted."""
        return self._is_fitted


def evaluate_calibration(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate probability calibration using reliability diagram metrics.
    
    Computes Expected Calibration Error (ECE) and Maximum Calibration
    Error (MCE) as well as data for plotting reliability diagrams.
    
    .. math::
        ECE = \\sum_{m=1}^{M} \\frac{|B_m|}{n} |acc(B_m) - conf(B_m)|
    
    .. math::
        MCE = \\max_{m \\in \\{1,...,M\\}} |acc(B_m) - conf(B_m)|
    
    Where B_m are bins, acc is accuracy, conf is mean confidence.
    
    Args:
        y_prob: Predicted probabilities, shape (n_samples,)
        y_true: True binary labels, shape (n_samples,)
        n_bins: Number of bins for calibration curve
    
    Returns:
        Tuple of (ECE, MCE, bin_accuracies, bin_confidences)
    
    Example:
        >>> ece, mce, accs, confs = evaluate_calibration(probs, labels)
        >>> print(f"ECE: {ece:.4f}, MCE: {mce:.4f}")
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        
        if i == n_bins - 1:  # Include right edge for last bin
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        
        bin_count = np.sum(in_bin)
        bin_counts[i] = bin_count
        
        if bin_count > 0:
            bin_accuracies[i] = np.mean(y_true[in_bin])
            bin_confidences[i] = np.mean(y_prob[in_bin])
    
    # Calculate metrics
    total_samples = len(y_prob)
    calibration_errors = np.abs(bin_accuracies - bin_confidences)
    
    # ECE: weighted average of calibration errors
    ece = np.sum((bin_counts / total_samples) * calibration_errors)
    
    # MCE: maximum calibration error (excluding empty bins)
    non_empty_errors = calibration_errors[bin_counts > 0]
    mce = np.max(non_empty_errors) if len(non_empty_errors) > 0 else 0.0
    
    return ece, mce, bin_accuracies, bin_confidences
