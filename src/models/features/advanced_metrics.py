"""
Advanced Basketball Metrics Calculator.

Implements Box Plus-Minus (BPM) and related metrics from box score data,
based on the methodologies documented by Spatial Jam and Basketball Reference.

BPM Formula (simplified):
    BPM = a1*Scoring + a2*Rebounding + a3*Assists + a4*Defense + a5*Turnovers
    
    Where coefficients are adjusted for position and role.

Key Metrics:
    - BPM: Box Plus-Minus (per 100 possessions relative to league average)
    - VORP: Value Over Replacement Player (cumulative BPM)
    - SOS: Strength of Schedule
    - SOS-Adjusted Win %: Win percentage adjusted for opponent strength

References:
    - Spatial Jam Glossary: https://spatialjam.com/glossary
    - Basketball Reference: https://www.basketball-reference.com/about/bpm2.html
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlayerBPM:
    """Container for player BPM calculation results."""
    player_id: str
    player_name: str
    team: str
    games_played: int
    minutes_per_game: float
    bpm: float
    offensive_bpm: float
    defensive_bpm: float
    vorp: float
    
    def __str__(self) -> str:
        return f"{self.player_name} ({self.team}): BPM={self.bpm:+.1f}, VORP={self.vorp:.1f}"


@dataclass
class TeamAdvancedMetrics:
    """Container for team-level advanced metrics."""
    team_code: str
    bpm_avg: float  # Average BPM of rotation players
    sos: float  # Strength of schedule (-1 to 1)
    sos_adjusted_win_pct: float  # SOS-normalized win percentage
    expected_wins: float  # Pythagorean expected wins
    
    def as_dict(self) -> Dict[str, float]:
        return {
            'bpm_avg': self.bpm_avg,
            'sos': self.sos,
            'sos_adjusted_win_pct': self.sos_adjusted_win_pct,
            'expected_wins': self.expected_wins,
        }


class AdvancedMetricsCalculator:
    """
    Calculate advanced basketball metrics from box score data.
    
    Implements BPM approximation using the simplified formula:
        BPM ≈ (Scoring + Rebounding + Playmaking + Defense) / Possessions
    
    Adjusted for playing time and team performance.
    
    Example:
        >>> calc = AdvancedMetricsCalculator()
        >>> bpm = calc.calculate_player_bpm(player_stats, team_stats, league_stats)
        >>> print(f"BPM: {bpm:+.1f}")
    """
    
    # BPM coefficients (simplified from Basketball Reference methodology)
    # These are approximations based on regression analysis
    BPM_COEFFICIENTS = {
        'pts_per_100': 0.025,      # Points contribution
        'trb_per_100': 0.035,      # Rebounds (offensive + defensive)
        'ast_per_100': 0.055,      # Assists (playmaking)
        'stl_per_100': 0.075,      # Steals (defensive disruption)
        'blk_per_100': 0.050,      # Blocks (rim protection)
        'tov_per_100': -0.065,     # Turnovers (negative)
        'pf_per_100': -0.020,      # Personal fouls (negative)
        'ts_pct': 0.15,            # True shooting efficiency
        'ast_to_tov': 0.02,        # Assist to turnover ratio
    }
    
    # Position adjustments (guards get less rebound credit, bigs get more)
    POSITION_ADJUSTMENTS = {
        'G': {'trb_per_100': 0.75, 'ast_per_100': 1.1, 'blk_per_100': 0.5},
        'F': {'trb_per_100': 1.0, 'ast_per_100': 1.0, 'blk_per_100': 1.0},
        'C': {'trb_per_100': 1.2, 'ast_per_100': 0.9, 'blk_per_100': 1.3},
    }
    
    # League average reference values (NBL 2023-24 approximations)
    # These are INDIVIDUAL PLAYER per-100-possession averages
    LEAGUE_AVERAGES = {
        'pts_per_100': 36.0,   # Individual player avg (~14 pts in 30 min)
        'trb_per_100': 14.0,   # Individual player avg (~5.5 reb in 30 min)
        'ast_per_100': 9.0,    # Individual player avg (~3.5 ast in 30 min)
        'pace': 72.0,          # Possessions per game (team level)
        'ts_pct': 0.56,
        'orb_pct': 0.26,
        'tov_pct': 0.14,
    }
    
    def __init__(
        self,
        league_averages: Optional[Dict[str, float]] = None,
        min_minutes_threshold: float = 10.0
    ):
        """
        Initialize the calculator.
        
        Args:
            league_averages: Override default league average values
            min_minutes_threshold: Minimum minutes per game for BPM calculation
        """
        self.league_avg = league_averages or self.LEAGUE_AVERAGES.copy()
        self.min_minutes = min_minutes_threshold
    
    def calculate_player_bpm(
        self,
        player_stats: pd.Series,
        team_stats: Optional[pd.Series] = None,
        position: str = 'F'
    ) -> float:
        """
        Calculate Box Plus-Minus for a player.
        
        BPM estimates how many points per 100 possessions a player
        contributes above a league-average player.
        
        Args:
            player_stats: Series with player box score stats:
                - pts, trb, ast, stl, blk, tov, pf, fgm, fga, fta, fg3m, mp
            team_stats: Optional team stats for context adjustment
            position: Player position (G/F/C) for coefficient adjustment
        
        Returns:
            BPM value (typically -10 to +10, 0 is league average)
        """
        # Extract stats with defaults
        pts = player_stats.get('pts', player_stats.get('points', 0))
        trb = player_stats.get('trb', player_stats.get('rebounds_total', 0))
        ast = player_stats.get('ast', player_stats.get('assists', 0))
        stl = player_stats.get('stl', player_stats.get('steals', 0))
        blk = player_stats.get('blk', player_stats.get('blocks', 0))
        tov = player_stats.get('tov', player_stats.get('turnovers', 0))
        pf = player_stats.get('pf', player_stats.get('personal_fouls', 0))
        fgm = player_stats.get('fgm', player_stats.get('field_goals_made', 0))
        fga = player_stats.get('fga', player_stats.get('field_goals_attempted', 0))
        fta = player_stats.get('fta', player_stats.get('free_throws_attempted', 0))
        fg3m = player_stats.get('fg3m', player_stats.get('three_pointers_made', 0))
        mp = player_stats.get('mp', player_stats.get('minutes', 0))
        
        if mp < self.min_minutes:
            return 0.0  # Not enough playing time for meaningful BPM
        
        # Calculate True Shooting %
        ts_pct = self._calculate_ts_pct(pts, fga, fta)
        
        # Calculate per-100-possession rates
        # Estimate possessions from minutes (team pace adjusted)
        poss_estimate = (mp / 40) * self.league_avg['pace']
        if poss_estimate <= 0:
            return 0.0
        
        per_100_multiplier = 100 / poss_estimate
        
        pts_per_100 = pts * per_100_multiplier
        trb_per_100 = trb * per_100_multiplier
        ast_per_100 = ast * per_100_multiplier
        stl_per_100 = stl * per_100_multiplier
        blk_per_100 = blk * per_100_multiplier
        tov_per_100 = tov * per_100_multiplier
        pf_per_100 = pf * per_100_multiplier
        
        # Get position adjustments
        pos_adj = self.POSITION_ADJUSTMENTS.get(position, self.POSITION_ADJUSTMENTS['F'])
        
        # Calculate raw BPM components
        scoring = (pts_per_100 - self.league_avg['pts_per_100']) * self.BPM_COEFFICIENTS['pts_per_100']
        rebounding = trb_per_100 * self.BPM_COEFFICIENTS['trb_per_100'] * pos_adj.get('trb_per_100', 1.0)
        playmaking = ast_per_100 * self.BPM_COEFFICIENTS['ast_per_100'] * pos_adj.get('ast_per_100', 1.0)
        steals = stl_per_100 * self.BPM_COEFFICIENTS['stl_per_100']
        blocks = blk_per_100 * self.BPM_COEFFICIENTS['blk_per_100'] * pos_adj.get('blk_per_100', 1.0)
        turnovers = tov_per_100 * self.BPM_COEFFICIENTS['tov_per_100']
        fouls = pf_per_100 * self.BPM_COEFFICIENTS['pf_per_100']
        
        # Efficiency bonus/penalty
        ts_diff = ts_pct - self.league_avg['ts_pct']
        efficiency = ts_diff * self.BPM_COEFFICIENTS['ts_pct'] * pts_per_100
        
        # Assist-to-turnover bonus
        ast_to_tov = ast / max(tov, 1)
        playmaking_bonus = max(0, ast_to_tov - 1.5) * self.BPM_COEFFICIENTS['ast_to_tov']
        
        # Sum components
        raw_bpm = (
            scoring + rebounding + playmaking + 
            steals + blocks + turnovers + fouls +
            efficiency + playmaking_bonus
        )
        
        # Apply team adjustment if available
        if team_stats is not None:
            team_adj = self._calculate_team_adjustment(team_stats)
            raw_bpm += team_adj * 0.15  # Weight team context
        
        # Clamp to reasonable range
        bpm = np.clip(raw_bpm, -15.0, 15.0)
        
        return float(bpm)
    
    def calculate_team_bpm(
        self,
        player_box_scores: pd.DataFrame,
        team_code: str,
        n_games: int = 10
    ) -> float:
        """
        Calculate weighted team BPM from roster.
        
        Args:
            player_box_scores: DataFrame with player stats
            team_code: Team identifier
            n_games: Number of recent games to consider
        
        Returns:
            Minutes-weighted average BPM for the team
        """
        team_players = player_box_scores[
            player_box_scores['team_code'] == team_code
        ].tail(n_games * 12)  # ~12 players per game
        
        if len(team_players) == 0:
            return 0.0
        
        # Aggregate player stats
        player_agg = team_players.groupby('player_id').agg({
            'points': 'mean',
            'rebounds_total': 'mean',
            'assists': 'mean',
            'steals': 'mean',
            'blocks': 'mean',
            'turnovers': 'mean',
            'personal_fouls': 'mean',
            'field_goals_made': 'mean',
            'field_goals_attempted': 'mean',
            'free_throws_attempted': 'mean',
            'three_pointers_made': 'mean',
            'minutes': 'sum',
        }).reset_index()
        
        # Calculate BPM for each player
        bpms = []
        weights = []
        
        for _, player in player_agg.iterrows():
            if player['minutes'] < self.min_minutes * n_games:
                continue
            
            # Map to expected format
            stats = pd.Series({
                'pts': player['points'],
                'trb': player['rebounds_total'],
                'ast': player['assists'],
                'stl': player['steals'],
                'blk': player['blocks'],
                'tov': player['turnovers'],
                'pf': player['personal_fouls'],
                'fgm': player['field_goals_made'],
                'fga': player['field_goals_attempted'],
                'fta': player['free_throws_attempted'],
                'fg3m': player['three_pointers_made'],
                'mp': player['minutes'] / n_games,
            })
            
            bpm = self.calculate_player_bpm(stats)
            bpms.append(bpm)
            weights.append(player['minutes'])
        
        if not bpms:
            return 0.0
        
        # Weighted average by minutes
        weighted_bpm = np.average(bpms, weights=weights)
        return float(weighted_bpm)
    
    def calculate_bpm_differential(
        self,
        home_team_bpm: float,
        away_team_bpm: float
    ) -> float:
        """
        Calculate BPM differential for a matchup.
        
        Args:
            home_team_bpm: Home team's weighted BPM
            away_team_bpm: Away team's weighted BPM
        
        Returns:
            BPM differential (positive favors home)
        """
        return home_team_bpm - away_team_bpm
    
    def calculate_sos(
        self,
        team_results: pd.DataFrame,
        team_code: str,
        all_team_ratings: Dict[str, float]
    ) -> float:
        """
        Calculate Strength of Schedule.
        
        SOS = Average opponent rating relative to league average.
        
        Args:
            team_results: DataFrame with game results
            team_code: Team to calculate SOS for
            all_team_ratings: Dict of team code -> rating
        
        Returns:
            SOS value (-1 to 1, 0 is average)
        """
        # Get team's games
        team_games = team_results[
            (team_results['home_team'] == team_code) |
            (team_results['away_team'] == team_code)
        ]
        
        if len(team_games) == 0:
            return 0.0
        
        # Get opponent ratings
        opponent_ratings = []
        
        for _, game in team_games.iterrows():
            opponent = game['away_team'] if game['home_team'] == team_code else game['home_team']
            rating = all_team_ratings.get(opponent, 0.0)
            opponent_ratings.append(rating)
        
        # Calculate average opponent strength
        league_avg_rating = np.mean(list(all_team_ratings.values()))
        avg_opponent = np.mean(opponent_ratings)
        
        # Normalize to -1 to 1 range
        max_diff = max(abs(r - league_avg_rating) for r in all_team_ratings.values()) or 1
        sos = (avg_opponent - league_avg_rating) / max_diff
        
        return float(np.clip(sos, -1.0, 1.0))
    
    def calculate_sos_adjusted_win_pct(
        self,
        win_pct: float,
        sos: float,
        adjustment_factor: float = 0.15
    ) -> float:
        """
        Adjust win percentage for strength of schedule.
        
        Teams with harder schedules get a boost, easier schedules get penalized.
        
        Args:
            win_pct: Raw win percentage (0 to 1)
            sos: Strength of schedule (-1 to 1)
            adjustment_factor: How much SOS affects the adjustment
        
        Returns:
            SOS-adjusted win percentage
        """
        # Positive SOS (hard schedule) increases adjusted win %
        adjustment = sos * adjustment_factor
        adjusted = win_pct + adjustment
        
        return float(np.clip(adjusted, 0.0, 1.0))
    
    def calculate_expected_wins(
        self,
        points_for: float,
        points_against: float,
        games_played: int,
        exponent: float = 13.91
    ) -> float:
        """
        Calculate Pythagorean expected wins.
        
        Uses the Pythagorean expectation formula:
            ExpWin% = PF^exp / (PF^exp + PA^exp)
        
        Args:
            points_for: Total points scored
            points_against: Total points allowed
            games_played: Number of games
            exponent: Pythagorean exponent (13.91 for basketball)
        
        Returns:
            Expected wins
        """
        if points_against <= 0 or games_played <= 0:
            return 0.0
        
        pf_exp = points_for ** exponent
        pa_exp = points_against ** exponent
        
        denominator = pf_exp + pa_exp
        if denominator == 0:
            return 0.0

        exp_win_pct = pf_exp / denominator
        exp_wins = exp_win_pct * games_played
        
        return float(exp_wins)
    
    def _calculate_ts_pct(
        self,
        points: float,
        fga: float,
        fta: float
    ) -> float:
        """Calculate True Shooting Percentage."""
        denominator = 2 * (fga + 0.44 * fta)
        # Handle zero or extremely small denominators to prevent infinity
        if denominator <= 1e-9:
            return 0.0
        return points / denominator
    
    def _calculate_team_adjustment(
        self,
        team_stats: pd.Series
    ) -> float:
        """Calculate team context adjustment for individual BPM."""
        # Good teams have players with inflated raw stats
        # This adjustment regresses to account for teammate quality
        
        team_margin = team_stats.get('point_diff', 0) / max(team_stats.get('games', 1), 1)
        
        # Normalize margin to BPM-like scale
        return team_margin / 10.0
    
    def calculate_team_advanced_metrics(
        self,
        team_code: str,
        team_results: pd.DataFrame,
        player_box_scores: Optional[pd.DataFrame] = None,
        all_team_ratings: Optional[Dict[str, float]] = None,
        n_games: int = 10
    ) -> TeamAdvancedMetrics:
        """
        Calculate all advanced metrics for a team.
        
        Args:
            team_code: Team identifier
            team_results: Game results DataFrame
            player_box_scores: Optional player stats for BPM calculation
            all_team_ratings: Dict of team ratings for SOS
            n_games: Number of recent games
        
        Returns:
            TeamAdvancedMetrics container
        """
        # Calculate BPM if player data available
        if player_box_scores is not None:
            bpm_avg = self.calculate_team_bpm(player_box_scores, team_code, n_games)
        else:
            bpm_avg = 0.0
        
        # Calculate SOS
        if all_team_ratings:
            sos = self.calculate_sos(team_results, team_code, all_team_ratings)
        else:
            sos = 0.0
        
        # Get team record
        team_games = team_results[
            (team_results['home_team'] == team_code) |
            (team_results['away_team'] == team_code)
        ].tail(n_games)
        
        if len(team_games) == 0:
            return TeamAdvancedMetrics(
                team_code=team_code,
                bpm_avg=0.0,
                sos=0.0,
                sos_adjusted_win_pct=0.5,
                expected_wins=0.0
            )
        
        # Calculate raw win %
        wins = 0
        points_for = 0
        points_against = 0
        
        for _, game in team_games.iterrows():
            is_home = game['home_team'] == team_code
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            
            if is_home:
                won = home_score > away_score
                points_for += home_score
                points_against += away_score
            else:
                won = away_score > home_score
                points_for += away_score
                points_against += home_score
            
            wins += int(won)
        
        win_pct = wins / len(team_games)
        
        # SOS-adjusted win %
        sos_adj_win_pct = self.calculate_sos_adjusted_win_pct(win_pct, sos)
        
        # Expected wins
        expected = self.calculate_expected_wins(points_for, points_against, len(team_games))
        
        return TeamAdvancedMetrics(
            team_code=team_code,
            bpm_avg=bpm_avg,
            sos=sos,
            sos_adjusted_win_pct=sos_adj_win_pct,
            expected_wins=expected
        )
