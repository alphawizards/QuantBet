
import pytest
from src.collectors.spatialjam_scraper import create_spatialjam_scraper, SpatialJamSubscriptionRequired, ShotRecord

class TestSpatialJamScraper:
    def test_raises_without_credentials(self):
        scraper = create_spatialjam_scraper()
        with pytest.raises(SpatialJamSubscriptionRequired):
            scraper.get_shot_machine()

    def test_mock_mode_returns_data(self):
        scraper = create_spatialjam_scraper(simulate=True)
        shots = scraper.get_shot_machine()

        assert len(shots) > 0
        assert isinstance(shots[0], ShotRecord)
        assert shots[0].shot_type in ['2pt', '3pt']

    def test_mock_data_validation(self):
        """Ensure mock data passes Pydantic validation."""
        scraper = create_spatialjam_scraper(simulate=True)
        shots = scraper.get_shot_machine()

        for shot in shots:
            assert -250 <= shot.x <= 250
            assert -52 <= shot.y <= 418
