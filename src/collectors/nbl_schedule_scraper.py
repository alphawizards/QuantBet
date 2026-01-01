"""
NBL Schedule Scraper

Scrapes upcoming games from NBL.com.au schedule page.
Based on browser analysis showing Webflow CMS structure.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ScheduledGame:
    """Scheduled NBL game from the schedule page."""
    game_id: str
    home_team: str
    away_team: str
    date_str: str  # e.g., "December 31, 2025"
    time_str: str  # e.g., "5:30 pm"
    venue: str
    round_number: Optional[int] = None
    game_url: str = ""
    
    @property
    def datetime_obj(self) -> Optional[datetime]:
        """Parse date and time into datetime object."""
        try:
            # Combine date and time strings
            combined = f"{self.date_str} {self.time_str}"
            # Parse with various formats
            formats = [
                "%B %d, %Y %I:%M %p",  # December 31, 2025 5:30 pm
                "%b %d, %Y %I:%M %p",   # Dec 31, 2025 5:30 pm
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(combined, fmt)
                except ValueError:
                    continue
            return None
        except Exception as e:
            logger.warning(f"Failed to parse datetime: {e}")
            return None


class NBLScheduleScraper:
    """Scraper for NBL.com.au schedule page."""
    
    BASE_URL = "https://www.nbl.com.au"
    SCHEDULE_URL = f"{BASE_URL}/schedule"
    
    # NBL team name mappings (full name -> code)
    TEAM_MAPPINGS = {
        "illawarra hawks": "ILL",
        "tasmania jackjumpers": "TAS",
        "melbourne united": "MEL",
        "sydney kings": "SYD",
        "perth wildcats": "PER",
        "brisbane bullets": "BRI",
        "adelaide 36ers": "ADL",
        "new zealand breakers": "NZB",
        "cairns taipans": "CAI",
        "south east melbourne phoenix": "SEM",
    }
    
    def __init__(self, timeout: int = 10):
        """
        Initialize scraper.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_team_code(self, team_name: str) -> str:
        """
        Convert full team name to code.
        
        Args:
            team_name: Full team name
            
        Returns:
            Team code (e.g., "MEL" for "Melbourne United")
        """
        normalized = team_name.lower().strip()
        return self.TEAM_MAPPINGS.get(normalized, team_name[:3].upper())
    
    def scrape_schedule(self, pages: int = 1) -> List[ScheduledGame]:
        """
        Scrape NBL schedule page(s).
        
        Args:
            pages: Number of pages to scrape (for pagination)
            
        Returns:
            List of scheduled games
        """
        games = []
        
        for page in range(1, pages + 1):
            url = self.SCHEDULE_URL
            if page > 1:
                url = f"{self.SCHEDULE_URL}?c4895a7e_page={page}"
            
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                page_games = self._parse_page(soup)
                games.extend(page_games)
                
                logger.info(f"Scraped {len(page_games)} games from page {page}")
                
            except Exception as e:
                logger.error(f"Failed to scrape page {page}: {e}")
                continue
        
        return games
    
    def _parse_page(self, soup: BeautifulSoup) -> List[ScheduledGame]:
        """Parse games from a schedule page."""
        games = []
        
        # Find all game links based on browser analysis
        game_links = soup.find_all('a', href=re.compile(r'^/games/'))
        
        for link in game_links:
            try:
                game_data = self._parse_game_element(link)
                if game_data:
                    games.append(game_data)
            except Exception as e:
                logger.warning(f"Failed to parse game element: {e}")
                continue
        
        return games
    
    def _parse_game_element(self, element) -> Optional[ScheduledGame]:
        """Parse individual game element."""
        # Extract game URL and ID
        game_url = element.get('href', '')
        game_id = game_url.split('/')[-1] if game_url else ""
        
        # Get all paragraph tags within the game element
        paragraphs = element.find_all('p')
        
        teams = []
        date_str = ""
        time_str = ""
        venue = ""
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            
            # Skip empty or very short text
            if len(text) < 3:
                continue
            
            # Identify component type based on content
            if self._looks_like_team_name(text):
                teams.append(text)
            elif self._looks_like_date(text):
                date_str = text
            elif self._looks_like_time(text):
                time_str = text
            elif self._looks_like_venue(text):
                venue = text
        
        # Validate we have minimum required data
        if len(teams) < 2 or not date_str:
            return None
        
        # Determine home/away based on position or venue
        home_team = teams[0]
        away_team = teams[1]
        
        # Extract round number from game ID if possible
        round_match = re.search(r'r(\d+)', game_id)
        round_number = int(round_match.group(1)) if round_match else None
        
        return ScheduledGame(
            game_id=game_id or f"unknown_{datetime.now().timestamp()}",
            home_team=home_team,
            away_team=away_team,
            date_str=date_str,
            time_str=time_str,
            venue=venue,
            round_number=round_number,
            game_url=f"{self.BASE_URL}{game_url}" if game_url else ""
        )
    
    def _looks_like_team_name(self, text: str) -> bool:
        """Check if text looks like a team name."""
        # Team names are typically multi-word, no numbers, no special chars
        return (
            len(text) > 5 and
            not re.search(r'\d', text) and
            not ':' in text and
            not ',' in text and
            text.lower() in ' '.join(self.TEAM_MAPPINGS.keys())
        )
    
    def _looks_like_date(self, text: str) -> bool:
        """Check if text looks like a date."""
        # Dates contain month name and day number
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december',
                  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        text_lower = text.lower()
        return any(month in text_lower for month in months) and ',' in text
    
    def _looks_like_time(self, text: str) -> bool:
        """Check if text looks like a time."""
        # Times contain colon and am/pm
        return ':' in text and ('am' in text.lower() or 'pm' in text.lower())
    
    def _looks_like_venue(self, text: str) -> bool:
        """Check if text looks like a venue name."""
        # Venues often contain "Centre", "Arena", "Stadium"
        venue_keywords = ['centre', 'center', 'arena', 'stadium', 'complex', 'court']
        return any(keyword in text.lower() for keyword in venue_keywords)
    
    def get_upcoming_games(self, days: int = 7) -> List[ScheduledGame]:
        """
        Get games scheduled in the next N days.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of upcoming games sorted by date
        """
        # Scrape multiple pages to ensure we get enough future games
        all_games = self.scrape_schedule(pages=3)
        
        now = datetime.now()
        cutoff = now + timedelta(days=days)
        
        upcoming = []
        for game in all_games:
            game_dt = game.datetime_obj
            if game_dt and now <= game_dt <= cutoff:
                upcoming.append(game)
        
        # Sort by date
        upcoming.sort(key=lambda g: g.datetime_obj or datetime.max)
        
        logger.info(f"Found {len(upcoming)} games in next {days} days")
        return upcoming


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = NBLScheduleScraper()
    games = scraper.get_upcoming_games(days=7)
    
    print(f"\nFound {len(games)} upcoming games:\n")
    for game in games:
        print(f"{game.home_team} vs {game.away_team}")
        print(f"  Date: {game.date_str} {game.time_str}")
        print(f"  Venue: {game.venue}")
        print(f"  URL: {game.game_url}\n")
