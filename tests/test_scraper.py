"""
Unit tests for NBL Data Scraper and Integration modules.
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

import pandas as pd


class TestNBLDataScraper:
    """Tests for the NBL data scraper."""
    
    def test_endpoint_urls_valid(self):
        """All endpoint URLs should be properly formatted."""
        from src.collectors.scraper import NBLDataScraper
        
        scraper = NBLDataScraper(use_cache=False)
        
        for key, url in scraper.ENDPOINTS.items():
            assert url.startswith("https://github.com/JaseZiv/nblr_data/releases/download/")
            assert url.endswith(".rds")
    
    def test_cache_directory_creation(self):
        """Cache directory should be created on initialization."""
        from src.collectors.scraper import NBLDataScraper
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            scraper = NBLDataScraper(cache_dir=str(cache_dir), use_cache=True)
            
            assert cache_dir.exists()
    
    def test_cache_path_generation(self):
        """Cache paths should be unique per endpoint."""
        from src.collectors.scraper import NBLDataScraper
        
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = NBLDataScraper(cache_dir=tmpdir, use_cache=True)
            
            path1 = scraper._get_cache_path("results_wide")
            path2 = scraper._get_cache_path("results_long")
            
            assert path1 != path2
            assert path1.suffix == ".parquet"
    
    @patch('src.collectors.scraper.requests.get')
    @patch('src.collectors.scraper.pyreadr.read_r')
    def test_download_rds_success(self, mock_pyreadr, mock_get):
        """Successful RDS download should return DataFrame."""
        from src.collectors.scraper import NBLDataScraper
        
        # Mock requests response
        mock_response = Mock()
        mock_response.iter_content = lambda chunk_size: [b'test data']
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Mock pyreadr result
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_pyreadr.return_value = {'data': test_df}
        
        scraper = NBLDataScraper(use_cache=False)
        result = scraper._download_rds("https://example.com/test.rds")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_clear_cache(self):
        """Clear cache should remove all parquet files."""
        from src.collectors.scraper import NBLDataScraper
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create some fake cache files
            (cache_dir / "test1.parquet").touch()
            (cache_dir / "test2.parquet").touch()
            
            scraper = NBLDataScraper(cache_dir=str(cache_dir), use_cache=True)
            deleted = scraper.clear_cache()
            
            assert deleted == 2
            assert not any(cache_dir.glob("*.parquet"))


class TestNBLDataIntegrator:
    """Tests for the NBL data integrator."""
    
    def test_team_name_normalization(self):
        """Team names should normalize to standard codes."""
        from src.collectors.integration import NBLDataIntegrator
        
        integrator = NBLDataIntegrator()
        
        # Test various formats
        assert integrator.normalize_team_name("Melbourne United") == "MEL"
        assert integrator.normalize_team_name("Sydney Kings") == "SYD"
        assert integrator.normalize_team_name("Perth Wildcats") == "PER"
        assert integrator.normalize_team_name("Wildcats") == "PER"
        assert integrator.normalize_team_name("perth wildcats") == "PER"  # Case insensitive
    
    def test_unknown_team_handling(self):
        """Unknown team names should return UNKNOWN."""
        from src.collectors.integration import NBLDataIntegrator
        
        integrator = NBLDataIntegrator(fuzzy_threshold=90)
        
        result = integrator.normalize_team_name("NonExistent Team XYZ")
        assert result == "UNKNOWN"
    
    def test_na_handling(self):
        """NaN values should be handled gracefully."""
        from src.collectors.integration import NBLDataIntegrator
        import numpy as np
        
        integrator = NBLDataIntegrator()
        
        assert integrator.normalize_team_name(None) == "UNKNOWN"
        assert integrator.normalize_team_name(np.nan) == "UNKNOWN"
    
    @pytest.fixture
    def sample_scraped_data(self):
        """Sample scraped data matching nblR format."""
        return pd.DataFrame({
            'match_id': [1, 2, 3],
            'match_time': ['2024-01-01 19:30:00', '2024-01-02 17:00:00', '2024-01-03 19:30:00'],
            'season': ['2023-2024', '2023-2024', '2023-2024'],
            'home_team_name': ['Melbourne United', 'Sydney Kings', 'Perth Wildcats'],
            'away_team_name': ['Sydney Kings', 'Perth Wildcats', 'Melbourne United'],
            'home_score_string': ['95', '88', '102'],
            'away_score_string': ['91', '92', '99'],
            'venue_name': ['John Cain Arena', 'Qudos Bank Arena', 'RAC Arena'],
            'attendance': [8500, 12000, 11000],
            'round_number': ['1', '1', '2'],
            'match_type': ['REGULAR', 'REGULAR', 'REGULAR'],
            'extra_periods_used': [0, 0, 0],
        })
    
    @pytest.fixture
    def sample_xlsx_data(self):
        """Sample xlsx data matching AusSportsBetting format."""
        return pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'Home Team': ['Melbourne United', 'Sydney Kings'],
            'Away Team': ['Sydney Kings', 'Perth Wildcats'],
            'Home Score': [95, 88],
            'Away Score': [91, 92],
            'Home Odds': [1.85, 2.10],
            'Away Odds': [2.00, 1.75],
            'Home Odds Open': [1.90, 2.05],
            'Home Odds Close': [1.85, 2.10],
            'Away Odds Open': [1.95, 1.80],
            'Away Odds Close': [2.00, 1.75],
        })
    
    def test_prepare_scraped_data(self, sample_scraped_data):
        """Scraped data should be properly prepared."""
        from src.collectors.integration import NBLDataIntegrator
        
        integrator = NBLDataIntegrator()
        result = integrator.prepare_scraped_data(sample_scraped_data)
        
        assert 'game_date' in result.columns
        assert 'home_team' in result.columns
        assert 'away_team' in result.columns
        assert result['home_team'].iloc[0] == 'MEL'
        assert result['away_team'].iloc[0] == 'SYD'
    
    def test_prepare_xlsx_data(self, sample_xlsx_data):
        """XLSX data should be properly prepared."""
        from src.collectors.integration import NBLDataIntegrator
        
        integrator = NBLDataIntegrator()
        result = integrator.prepare_xlsx_data(sample_xlsx_data)
        
        assert 'game_date' in result.columns
        assert 'home_odds' in result.columns
        assert 'away_odds' in result.columns
        assert result['home_team'].iloc[0] == 'MEL'
    
    def test_merge_stats(self, sample_scraped_data, sample_xlsx_data):
        """Merge stats should correctly count overlapping games."""
        from src.collectors.integration import NBLDataIntegrator
        
        integrator = NBLDataIntegrator()
        stats = integrator.get_merge_stats(sample_scraped_data, sample_xlsx_data)
        
        assert stats['total_scraped'] == 3
        assert stats['total_xlsx'] == 2
        assert stats['overlap'] == 2  # Two games match
        assert stats['scraped_only'] == 1  # Third game is scraped-only


class TestIntegration:
    """Integration tests for scraper and integration modules."""
    
    def test_imports(self):
        """All modules should import correctly."""
        from src.collectors import NBLDataScraper, NBLDataIntegrator
        
        assert NBLDataScraper is not None
        assert NBLDataIntegrator is not None
    
    def test_scraper_initialization(self):
        """Scraper should initialize with default settings."""
        from src.collectors import NBLDataScraper
        
        scraper = NBLDataScraper()
        
        assert scraper.use_cache == True
        assert scraper.cache_ttl_hours == 24
