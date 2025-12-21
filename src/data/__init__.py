"""
NBL Data Module.

Provides scraping and integration of NBL basketball data
from multiple sources including nblR and xlsx files.
"""

from .scraper import NBLDataScraper
from .integration import NBLDataIntegrator

__all__ = [
    "NBLDataScraper",
    "NBLDataIntegrator",
]
