"""Export utilities for the candidate catalog.

Supports CSV and FITS export formats for interoperability with
standard astronomy tools (TOPCAT, DS9, etc.).

TODO: Students can add VOTable export using astropy.io.votable.
"""

from pathlib import Path

import pandas as pd

from exohunter import config
from exohunter.catalog.candidates import CandidateCatalog
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def export_to_csv(
    catalog: CandidateCatalog,
    output_path: Path | None = None,
) -> Path:
    """Export the candidate catalog to a CSV file.

    Args:
        catalog: The ``CandidateCatalog`` to export.
        output_path: Destination file path. Defaults to
            ``data/output/candidates.csv``.

    Returns:
        The ``Path`` to the written CSV file.
    """
    if output_path is None:
        output_path = config.OUTPUT_DIR / "candidates.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = catalog.to_dataframe()
    df.to_csv(output_path, index=False)

    logger.info("Exported %d candidates to %s", len(catalog), output_path)
    return output_path


def export_to_fits(
    catalog: CandidateCatalog,
    output_path: Path | None = None,
) -> Path:
    """Export the candidate catalog to a FITS table.

    Args:
        catalog: The ``CandidateCatalog`` to export.
        output_path: Destination file path. Defaults to
            ``data/output/candidates.fits``.

    Returns:
        The ``Path`` to the written FITS file.
    """
    from astropy.table import Table

    if output_path is None:
        output_path = config.OUTPUT_DIR / "candidates.fits"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = catalog.to_dataframe()
    table = Table.from_pandas(df)
    table.write(output_path, format="fits", overwrite=True)

    logger.info("Exported %d candidates to %s (FITS)", len(catalog), output_path)
    return output_path
