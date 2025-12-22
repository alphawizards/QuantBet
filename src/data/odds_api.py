"""
The Odds API Client for NBL/WNBL Odds.

Fetches live odds from Australian bookmakers for NBL games.
Uses the free tier (500 requests/month).

API Documentation: https://the-odds-api.com/liveapi/guides/v4/
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class GameOdds:
    """Odds for a single game from The Odds API."""
    event_id: str
    sport: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: List[Dict]
    
    # Best odds across all bookmakers
    best_home_odds: float = 0.0
    best_away_odds: float = 0.0
    best_home_bookmaker: str = ""
    best_away_bookmaker: str = ""


@dataclass 
class APIQuotaInfo:
    """API quota usage information."""
    requests_used: int
    requests_remaining: int


class OddsAPIClient:
    """
    Client for The Odds API.
    
    Fetches live odds for NBL and WNBL games from Australian bookmakers.
    
    Example:
        >>> client = OddsAPIClient()
        >>> odds = client.get_nbl_odds()
        >>> for game in odds:
        ...     print(f"{game.home_team} vs {game.away_team}")
        ...     print(f"Best home odds: {game.best_home_odds} ({game.best_home_bookmaker})")
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Sport keys for Australian basketball
    SPORTS = {
        "nbl": "basketball_nbl",
        "wnbl": "basketball_wnbl",
    }
    
    # Australian bookmakers
    DEFAULT_REGION = "au"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: str = DEFAULT_REGION,
    ):
        """
        Initialize the Odds API client.
        
        Args:
            api_key: The Odds API key. If None, reads from ODDS_API_KEY env var.
            region: Bookmaker region (default: 'au' for Australia).
        """
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not set. Get one at https://the-odds-api.com/")
        
        self.region = region
        self.last_quota: Optional[APIQuotaInfo] = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _make_request(self, endpoint: str, params: Dict) -> tuple:
        """
        Make an API request with retry logic.
        
        Returns:
            Tuple of (response_json, quota_info)
        """
        params["apiKey"] = self.api_key
        
        logger.info(f"Requesting: {endpoint}")
        response = requests.get(
            f"{self.BASE_URL}{endpoint}",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        # Extract quota info from headers
        requests_used = response.headers.get("x-requests-used", 0)
        requests_remaining = response.headers.get("x-requests-remaining", 0)
        
        self.last_quota = APIQuotaInfo(
            requests_used=int(requests_used),
            requests_remaining=int(requests_remaining)
        )
        
        logger.info(
            f"API quota: {self.last_quota.requests_used} used, "
            f"{self.last_quota.requests_remaining} remaining"
        )
        
        return response.json(), self.last_quota
    
    def get_sports(self) -> List[Dict]:
        """
        Get list of available sports.
        
        Returns:
            List of sport objects with keys: key, group, title, active
        """
        data, _ = self._make_request("/sports/", {})
        return data
    
    def get_odds(
        self,
        sport: str,
        markets: str = "h2h",
        odds_format: str = "decimal"
    ) -> List[GameOdds]:
        """
        Get odds for a specific sport.
        
        Args:
            sport: Sport key (e.g., 'basketball_nbl')
            markets: Markets to fetch (h2h, spreads, totals)
            odds_format: 'decimal' or 'american'
        
        Returns:
            List of GameOdds objects
        """
        params = {
            "regions": self.region,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        
        data, _ = self._make_request(f"/sports/{sport}/odds/", params)
        
        games = []
        for event in data:
            game = self._parse_event(event, sport)
            games.append(game)
        
        return games
    
    def get_nbl_odds(self) -> List[GameOdds]:
        """Get current NBL odds from Australian bookmakers."""
        return self.get_odds(self.SPORTS["nbl"])
    
    def get_wnbl_odds(self) -> List[GameOdds]:
        """Get current WNBL odds from Australian bookmakers."""
        return self.get_odds(self.SPORTS["wnbl"])
    
    def get_all_basketball_odds(self) -> Dict[str, List[GameOdds]]:
        """
        Get odds for both NBL and WNBL.
        
        Returns:
            Dictionary with 'nbl' and 'wnbl' keys
        """
        return {
            "nbl": self.get_nbl_odds(),
            "wnbl": self.get_wnbl_odds(),
        }
    
    def _parse_event(self, event: Dict, sport: str) -> GameOdds:
        """Parse an event from the API response into a GameOdds object."""
        
        bookmakers = event.get("bookmakers", [])
        
        # Find best odds for each outcome
        best_home_odds = 0.0
        best_away_odds = 0.0
        best_home_bookmaker = ""
        best_away_bookmaker = ""
        
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        for bookmaker in bookmakers:
            bookmaker_name = bookmaker.get("title", "")
            
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        team = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        
                        if team == home_team and price > best_home_odds:
                            best_home_odds = price
                            best_home_bookmaker = bookmaker_name
                        elif team == away_team and price > best_away_odds:
                            best_away_odds = price
                            best_away_bookmaker = bookmaker_name
        
        # Parse commence time
        commence_str = event.get("commence_time", "")
        try:
            commence_time = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            commence_time = datetime.now()
        
        return GameOdds(
            event_id=event.get("id", ""),
            sport=sport,
            commence_time=commence_time,
            home_team=home_team,
            away_team=away_team,
            bookmakers=bookmakers,
            best_home_odds=best_home_odds,
            best_away_odds=best_away_odds,
            best_home_bookmaker=best_home_bookmaker,
            best_away_bookmaker=best_away_bookmaker,
        )
    
    def get_quota(self) -> Optional[APIQuotaInfo]:
        """Get the last known API quota information."""
        return self.last_quota


# Convenience function for quick access
def fetch_nbl_odds(api_key: Optional[str] = None) -> List[GameOdds]:
    """
    Fetch current NBL odds.
    
    Args:
        api_key: Optional API key (defaults to ODDS_API_KEY env var)
    
    Returns:
        List of GameOdds for upcoming NBL games
    """
    client = OddsAPIClient(api_key=api_key)
    return client.get_nbl_odds()
