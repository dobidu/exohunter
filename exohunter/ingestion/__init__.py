"""Data ingestion layer — concurrent download of TESS light curves."""

from exohunter.ingestion.downloader import (
    download_batch,
    download_lightcurve,
    search_targets,
)

__all__ = ["search_targets", "download_lightcurve", "download_batch"]
