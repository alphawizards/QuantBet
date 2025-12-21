"""
SQLAlchemy ORM Models for QuantBet NBL/WNBL Betting System.

These models mirror the PostgreSQL schema and provide Python-native
access to the database with proper type hints.
"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Optional, List
from uuid import UUID, uuid4

from sqlalchemy import (
    String, Integer, Boolean, Text, Numeric, DateTime,
    ForeignKey, CheckConstraint, UniqueConstraint, Index,
    Enum, JSON, func
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB


# ============================================================================
# Enum Definitions
# ============================================================================

class LeagueType(PyEnum):
    """Australian basketball leagues."""
    NBL = "NBL"
    WNBL = "WNBL"


class GameStatus(PyEnum):
    """Possible game states."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class DNPReason(PyEnum):
    """Reasons a player did not play."""
    INJURY = "injury"
    REST = "rest"
    COACH_DECISION = "coach_decision"
    PERSONAL = "personal"
    SUSPENSION = "suspension"
    NOT_WITH_TEAM = "not_with_team"


class BetType(PyEnum):
    """Types of bets supported."""
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"


class BetOutcome(PyEnum):
    """Possible bet outcomes."""
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


# ============================================================================
# Base Model
# ============================================================================

class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ============================================================================
# Team Model
# ============================================================================

class Team(Base):
    """
    NBL/WNBL team reference data.
    
    Includes geographic coordinates for travel distance calculations
    in the feature engineering pipeline.
    """
    __tablename__ = "teams"
    
    team_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    team_code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    team_name: Mapped[str] = mapped_column(String(100), nullable=False)
    league: Mapped[LeagueType] = mapped_column(Enum(LeagueType), nullable=False)
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    state: Mapped[Optional[str]] = mapped_column(String(50))
    country: Mapped[str] = mapped_column(String(50), default="Australia")
    venue_name: Mapped[Optional[str]] = mapped_column(String(100))
    latitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 7))
    longitude: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 7))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    home_games: Mapped[List["Game"]] = relationship(
        back_populates="home_team",
        foreign_keys="Game.home_team_id"
    )
    away_games: Mapped[List["Game"]] = relationship(
        back_populates="away_team",
        foreign_keys="Game.away_team_id"
    )
    players: Mapped[List["Player"]] = relationship(back_populates="current_team")
    
    def __repr__(self) -> str:
        return f"<Team(code={self.team_code}, name={self.team_name})>"


# ============================================================================
# Game Model
# ============================================================================

class Game(Base):
    """
    NBL/WNBL game metadata.
    
    Stores scheduling info, venue details, and final scores.
    Quarter-by-quarter scores are tracked for momentum analysis.
    """
    __tablename__ = "games"
    
    game_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    external_game_id: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    league: Mapped[LeagueType] = mapped_column(Enum(LeagueType), nullable=False)
    season: Mapped[str] = mapped_column(String(10), nullable=False)
    round_number: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Scheduling
    scheduled_datetime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    actual_start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    actual_end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Teams
    home_team_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("teams.team_id"),
        nullable=False
    )
    away_team_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("teams.team_id"),
        nullable=False
    )
    
    # Venue
    venue_name: Mapped[Optional[str]] = mapped_column(String(100))
    venue_city: Mapped[Optional[str]] = mapped_column(String(50))
    venue_country: Mapped[Optional[str]] = mapped_column(String(50), default="Australia")
    is_neutral_site: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Results
    status: Mapped[GameStatus] = mapped_column(
        Enum(GameStatus),
        default=GameStatus.SCHEDULED
    )
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Quarter scores
    home_q1_score: Mapped[Optional[int]] = mapped_column(Integer)
    home_q2_score: Mapped[Optional[int]] = mapped_column(Integer)
    home_q3_score: Mapped[Optional[int]] = mapped_column(Integer)
    home_q4_score: Mapped[Optional[int]] = mapped_column(Integer)
    home_ot_score: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    away_q1_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_q2_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_q3_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_q4_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_ot_score: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    
    # Metadata
    attendance: Mapped[Optional[int]] = mapped_column(Integer)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    home_team: Mapped["Team"] = relationship(
        back_populates="home_games",
        foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(
        back_populates="away_games",
        foreign_keys=[away_team_id]
    )
    player_stats: Mapped[List["PlayerStats"]] = relationship(back_populates="game")
    odds_history: Mapped[List["OddsHistory"]] = relationship(back_populates="game")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("home_team_id != away_team_id", name="different_teams"),
        Index("idx_games_datetime", "scheduled_datetime"),
        Index("idx_games_season", "league", "season"),
    )
    
    def __repr__(self) -> str:
        return f"<Game(id={self.game_id}, date={self.scheduled_datetime.date()})>"


# ============================================================================
# Player Model
# ============================================================================

class Player(Base):
    """
    NBL/WNBL player reference data.
    
    Tracks import status which is critical for NBL analysis as import
    players (typically 2-3 per team) often have outsized impact.
    """
    __tablename__ = "players"
    
    player_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    external_player_id: Mapped[Optional[str]] = mapped_column(String(50), unique=True)
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Current team
    current_team_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("teams.team_id")
    )
    jersey_number: Mapped[Optional[str]] = mapped_column(String(5))
    position: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Import status (critical for NBL)
    is_import: Mapped[bool] = mapped_column(Boolean, default=False)
    import_tier: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Physical attributes
    height_cm: Mapped[Optional[int]] = mapped_column(Integer)
    weight_kg: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    birth_date: Mapped[Optional[date]] = mapped_column(DateTime)
    nationality: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    current_team: Mapped[Optional["Team"]] = relationship(back_populates="players")
    stats: Mapped[List["PlayerStats"]] = relationship(back_populates="player")
    
    @property
    def display_name(self) -> str:
        """Full display name."""
        return f"{self.first_name} {self.last_name}"
    
    def __repr__(self) -> str:
        return f"<Player(name={self.display_name}, import={self.is_import})>"


# ============================================================================
# Player Stats Model
# ============================================================================

class PlayerStats(Base):
    """
    Player box score statistics.
    
    Handles FIBA 40-minute games with per-40-minute scaling.
    Properly tracks DNP (Did Not Play) cases with reasons.
    
    Per-40-minute stats are calculated via database trigger or
    application layer:
        stat_per_40 = stat * (40 * 60) / seconds_played
    """
    __tablename__ = "player_stats"
    
    stat_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    game_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("games.game_id", ondelete="CASCADE"),
        nullable=False
    )
    player_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("players.player_id"),
        nullable=False
    )
    team_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("teams.team_id"),
        nullable=False
    )
    
    # Playing time (in seconds)
    seconds_played: Mapped[int] = mapped_column(Integer, default=0)
    
    # DNP handling
    did_not_play: Mapped[bool] = mapped_column(Boolean, default=False)
    dnp_reason: Mapped[Optional[DNPReason]] = mapped_column(Enum(DNPReason))
    
    # Starter status
    is_starter: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    
    # Scoring
    points: Mapped[int] = mapped_column(Integer, default=0)
    field_goals_made: Mapped[int] = mapped_column(Integer, default=0)
    field_goals_attempted: Mapped[int] = mapped_column(Integer, default=0)
    three_pointers_made: Mapped[int] = mapped_column(Integer, default=0)
    three_pointers_attempted: Mapped[int] = mapped_column(Integer, default=0)
    free_throws_made: Mapped[int] = mapped_column(Integer, default=0)
    free_throws_attempted: Mapped[int] = mapped_column(Integer, default=0)
    
    # Rebounding
    offensive_rebounds: Mapped[int] = mapped_column(Integer, default=0)
    defensive_rebounds: Mapped[int] = mapped_column(Integer, default=0)
    
    # Playmaking & Defense
    assists: Mapped[int] = mapped_column(Integer, default=0)
    steals: Mapped[int] = mapped_column(Integer, default=0)
    blocks: Mapped[int] = mapped_column(Integer, default=0)
    turnovers: Mapped[int] = mapped_column(Integer, default=0)
    personal_fouls: Mapped[int] = mapped_column(Integer, default=0)
    
    # Advanced
    plus_minus: Mapped[Optional[int]] = mapped_column(Integer)
    efficiency_rating: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    
    # Per-40-minute stats (computed)
    pts_per_40: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    reb_per_40: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    ast_per_40: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    # Relationships
    game: Mapped["Game"] = relationship(back_populates="player_stats")
    player: Mapped["Player"] = relationship(back_populates="stats")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="unique_player_game"),
        CheckConstraint("field_goals_made <= field_goals_attempted", name="valid_fg"),
        CheckConstraint("three_pointers_made <= three_pointers_attempted", name="valid_3pt"),
        CheckConstraint("free_throws_made <= free_throws_attempted", name="valid_ft"),
        Index("idx_player_stats_game", "game_id"),
        Index("idx_player_stats_player", "player_id", "game_id"),
    )
    
    @property
    def total_rebounds(self) -> int:
        """Total rebounds (offensive + defensive)."""
        return self.offensive_rebounds + self.defensive_rebounds
    
    @property
    def minutes_played(self) -> float:
        """Playing time in minutes."""
        return self.seconds_played / 60.0
    
    def calculate_per_40_stats(self) -> None:
        """Calculate per-40-minute statistics for FIBA scaling."""
        if self.seconds_played > 0:
            scale = 2400.0 / self.seconds_played  # 40 * 60 = 2400 seconds
            self.pts_per_40 = Decimal(str(round(self.points * scale, 2)))
            self.reb_per_40 = Decimal(str(round(self.total_rebounds * scale, 2)))
            self.ast_per_40 = Decimal(str(round(self.assists * scale, 2)))
        else:
            self.pts_per_40 = Decimal("0")
            self.reb_per_40 = Decimal("0")
            self.ast_per_40 = Decimal("0")
    
    def __repr__(self) -> str:
        return f"<PlayerStats(player={self.player_id}, pts={self.points})>"


# ============================================================================
# Odds History Model
# ============================================================================

class OddsHistory(Base):
    """
    Betting odds history tracking.
    
    Tracks lines from opening to closing with implied probabilities.
    Supports multiple sportsbooks for line shopping analysis.
    
    Key design decisions:
        1. Store decimal odds (Australian standard)
        2. Auto-calculate implied probability: 1 / decimal_odds
        3. Mark opening and closing lines explicitly
        4. Track line movement via recorded_at timestamp
    """
    __tablename__ = "odds_history"
    
    odds_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    game_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("games.game_id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Sportsbook
    sportsbook: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Bet details
    bet_type: Mapped[BetType] = mapped_column(Enum(BetType), nullable=False)
    selection: Mapped[str] = mapped_column(String(100), nullable=False)
    line_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 2))
    
    # Odds
    decimal_odds: Mapped[Decimal] = mapped_column(Numeric(8, 3), nullable=False)
    
    # Line timing
    is_opening_line: Mapped[bool] = mapped_column(Boolean, default=False)
    is_closing_line: Mapped[bool] = mapped_column(Boolean, default=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp()
    )
    
    # Outcome
    outcome: Mapped[BetOutcome] = mapped_column(
        Enum(BetOutcome),
        default=BetOutcome.PENDING
    )
    
    # Raw data storage
    raw_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp()
    )
    
    # Relationships
    game: Mapped["Game"] = relationship(back_populates="odds_history")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("decimal_odds >= 1.0", name="valid_odds"),
        Index("idx_odds_game", "game_id", "bet_type"),
        Index("idx_odds_sportsbook", "sportsbook", "game_id"),
    )
    
    @property
    def implied_probability(self) -> Decimal:
        """
        Calculate implied probability from decimal odds.
        
        Formula: p = 1 / decimal_odds
        
        Note: This does NOT remove the vig/overround. For true probability
        estimation, vig must be removed separately.
        """
        return Decimal("1") / self.decimal_odds
    
    @property
    def american_odds(self) -> int:
        """Convert decimal odds to American format."""
        if self.decimal_odds >= 2.0:
            return int((self.decimal_odds - 1) * 100)
        else:
            return int(-100 / (self.decimal_odds - 1))
    
    def __repr__(self) -> str:
        return f"<OddsHistory(game={self.game_id}, odds={self.decimal_odds})>"
