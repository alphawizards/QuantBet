"""
Bet Tracking API Endpoints

Allows users to track their actual placed bets and monitor real P/L performance.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime, timezone
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.core.database.connection import get_db

router = APIRouter(prefix="/api/bets", tags=["Bet Tracking"])


# ============================================================================
# Pydantic Models
# ============================================================================

class TrackBetRequest(BaseModel):
    """Request to track a new bet"""
    game_id: str = Field(..., description="Unique game identifier")
    home_team: str
    away_team: str
    game_date: datetime
    bet_on: str = Field(..., description="HOME or AWAY")
    prediction: float = Field(..., ge=0, le=1)
    odds: float = Field(..., gt=1.0)
    stake: float = Field(..., gt=0)
    edge: Optional[float] = None
    model_id: Optional[str] = None
    confidence: Optional[str] = Field(None, pattern="^(HIGH|MEDIUM|LOW)$")
    bookmaker: Optional[str] = None
    notes: Optional[str] = None
    
    @validator('bet_on')
    def validate_bet_on(cls, v):
        if v.upper() not in ['HOME', 'AWAY']:
            raise ValueError('bet_on must be HOME or AWAY')
        return v.upper()


class UpdateBetResultRequest(BaseModel):
    """Request to update bet outcome"""
    actual_result: str = Field(..., description="HOME or AWAY")
    status: str = Field(default="WON", pattern="^(WON|LOST|VOID)$")
    
    @validator('actual_result')
    def validate_actual_result(cls, v):
        if v.upper() not in ['HOME', 'AWAY']:
            raise ValueError('actual_result must be HOME or AWAY')
        return v.upper()


class TrackedBet(BaseModel):
    """Response model for a tracked bet"""
    id: int
    bet_id: str
    user_id: str
    game_id: str
    home_team: str
    away_team: str
    game_date: datetime
    bet_on: str
    prediction: float
    odds: float
    stake: float
    edge: Optional[float]
    model_id: Optional[str]
    confidence: Optional[str]
    bookmaker: Optional[str]
    status: str
    actual_result: Optional[str]
    profit: Optional[float]
    settled_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    notes: Optional[str]


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_profit(stake: float, odds: float, won: bool) -> float:
    """Calculate profit/loss for a bet"""
    if won:
        return stake * (odds - 1)  # Profit = stake * (odds - 1)
    else:
        return -stake  # Loss = -(stake)


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/track", response_model=TrackedBet, status_code=201)
async def track_bet(
    bet: TrackBetRequest,
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db)
):
    """
    Track a new bet
    
    Logs a bet placement for future monitoring and P/L tracking.
    """
    try:
        bet_id = str(uuid.uuid4())
        
        # Normalize datetime to prevent timezone-aware/naive conflicts
        # Convert to UTC and strip timezone info for database storage
        game_date_normalized = bet.game_date
        if game_date_normalized.tzinfo is not None:
            # Convert to UTC and remove timezone info
            game_date_normalized = game_date_normalized.astimezone(timezone.utc).replace(tzinfo=None)
        
        query = text("""
            INSERT INTO user_bets (
                bet_id, user_id, game_id, home_team, away_team, game_date,
                bet_on, prediction, odds, stake, edge, model_id, confidence,
                bookmaker, notes, status
            )
            VALUES (:bet_id, :user_id, :game_id, :home_team, :away_team, :game_date,
                    :bet_on, :prediction, :odds, :stake, :edge, :model_id, :confidence,
                    :bookmaker, :notes, 'PENDING')
            RETURNING *
        """)
        
        result = await db.execute(query, {
            "bet_id": bet_id,
            "user_id": user_id,
            "game_id": bet.game_id,
            "home_team": bet.home_team,
            "away_team": bet.away_team,
            "game_date": game_date_normalized,  # Use normalized datetime
            "bet_on": bet.bet_on,
            "prediction": bet.prediction,
            "odds": bet.odds,
            "stake": bet.stake,
            "edge": bet.edge,
            "model_id": bet.model_id,
            "confidence": bet.confidence,
            "bookmaker": bet.bookmaker,
            "notes": bet.notes
        })
        
        row = result.fetchone()
        await db.commit()
        
        # Convert row to dict
        bet_data = dict(row._mapping)
        return TrackedBet(**bet_data)
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to track bet: {str(e)}")


@router.get("/recent", response_model=List[TrackedBet])
async def get_recent_bets(
    limit: int = 20,
    status: Optional[str] = None,
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent tracked bets
    
    Returns most recent bets, optionally filtered by status (PENDING, WON, LOST, etc.)
    """
    try:
        if status:
            query = text("""
                SELECT * FROM user_bets
                WHERE user_id = :user_id AND status = :status
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = await db.execute(query, {
                "user_id": user_id,
                "status": status.upper(),
                "limit": limit
            })
        else:
            query = text("""
                SELECT * FROM user_bets
                WHERE user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :limit
            """)
            result = await db.execute(query, {
                "user_id": user_id,
                "limit": limit
            })
        
        rows = result.fetchall()
        bets = [TrackedBet(**dict(row._mapping)) for row in rows]
        return bets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch bets: {str(e)}")


@router.put("/{bet_id}/result", response_model=TrackedBet)
async def update_bet_result(
    bet_id: str,
    result_data: UpdateBetResultRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update the outcome of a bet
    
    Marks a pending bet as won/lost and calculates profit.
    """
    try:
        # First get the current bet
        query = text("""
            SELECT bet_on, stake, odds FROM user_bets
            WHERE bet_id = :bet_id
        """)
        result = await db.execute(query, {"bet_id": bet_id})
        bet_data = result.fetchone()
        
        if not bet_data:
            raise HTTPException(status_code=404, detail="Bet not found")
        
        bet_on, stake, odds = bet_data
        
        # Determine if bet won
        won = (bet_on == result_data.actual_result)
        
        # Calculate profit
        profit = calculate_profit(float(stake), float(odds), won)
        
        # Update status based on win/loss
        final_status = result_data.status if result_data.status == "VOID" else ("WON" if won else "LOST")
        
        # Update the bet
        update_query = text("""
            UPDATE user_bets
            SET 
                status = :status,
                actual_result = :actual_result,
                profit = :profit,
                settled_at = CURRENT_TIMESTAMP
            WHERE bet_id = :bet_id
            RETURNING *
        """)
        
        updated_result = await db.execute(update_query, {
            "status": final_status,
            "actual_result": result_data.actual_result,
            "profit": profit,
            "bet_id": bet_id
        })
        
        updated_row = updated_result.fetchone()
        await db.commit()
        
        return TrackedBet(**dict(updated_row._mapping))
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update bet: {str(e)}")


@router.get("/stats", response_model=dict)
async def get_bet_stats(
    user_id: str = "default_user",
    db: AsyncSession = Depends(get_db)
):
    """
    Get betting statistics
    
    Returns overall profit, ROI, win rate, etc. for tracked bets.
    """
    try:
        query = text("""
            SELECT 
                COUNT(*) as total_bets,
                COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending_bets,
                COUNT(CASE WHEN status = 'WON' THEN 1 END) as won_bets,
                COUNT(CASE WHEN status = 'LOST' THEN 1 END) as lost_bets,
                COALESCE(SUM(stake), 0) as total_staked,
                COALESCE(SUM(profit), 0) as total_profit,
                COALESCE(SUM(CASE WHEN status IN ('WON', 'LOST') THEN stake ELSE 0 END), 0) as settled_stake
            FROM user_bets
            WHERE user_id = :user_id
        """)
        
        result = await db.execute(query, {"user_id": user_id})
        row = result.fetchone()
        
        total_bets, pending, won, lost, total_staked, total_profit, settled_stake = row
        
        # Calculate win rate and ROI
        settled_bets = won + lost
        win_rate = (won / settled_bets * 100) if settled_bets > 0 else 0
        roi = (float(total_profit) / float(settled_stake) * 100) if settled_stake > 0 else 0
        
        return {
            "total_bets": total_bets,
            "pending_bets": pending,
            "won_bets": won,
            "lost_bets": lost,
            "win_rate": round(win_rate, 2),
            "total_staked": float(total_staked),
            "total_profit": float(total_profit),
            "roi": round(roi, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")
