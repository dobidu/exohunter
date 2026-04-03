"""Export utilities for the candidate catalog.

Supports CSV, FITS, and VOTable export formats for interoperability
with standard astronomy tools (TOPCAT, Aladin, DS9, etc.) and the
International Virtual Observatory Alliance (IVOA) ecosystem.
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


def export_to_votable(
    catalog: CandidateCatalog,
    output_path: Path | None = None,
) -> Path:
    """Export the candidate catalog to a VOTable (XML) file.

    VOTable is the standard data format of the International Virtual
    Observatory Alliance (IVOA).  The exported file can be loaded
    directly into TOPCAT, Aladin, or any VO-compatible tool.

    Column metadata (UCD, unit, description) follows IVOA conventions
    so that VO tools can automatically recognise the physical meaning
    of each column.

    Args:
        catalog: The ``CandidateCatalog`` to export.
        output_path: Destination file path. Defaults to
            ``data/output/candidates.votable.xml``.

    Returns:
        The ``Path`` to the written VOTable file.
    """
    from astropy.table import Table

    if output_path is None:
        output_path = config.OUTPUT_DIR / "candidates.votable.xml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = catalog.to_dataframe()
    table = Table.from_pandas(df)

    # Add IVOA-standard metadata to columns.
    # UCDs (Unified Content Descriptors) let VO tools identify
    # what each column represents without parsing column names.
    _column_metadata = {
        "tic_id": {
            "ucd": "meta.id",
            "description": "TESS Input Catalog identifier",
        },
        "period": {
            "ucd": "time.period",
            "unit": "d",
            "description": "Orbital period",
        },
        "epoch": {
            "ucd": "time.epoch",
            "unit": "d",
            "description": "Mid-transit time (BTJD)",
        },
        "duration": {
            "ucd": "time.duration",
            "unit": "d",
            "description": "Transit duration",
        },
        "depth": {
            "ucd": "phot.flux;arith.ratio",
            "description": "Fractional transit depth",
        },
        "snr": {
            "ucd": "stat.snr",
            "description": "Signal-to-noise ratio",
        },
        "bls_power": {
            "ucd": "stat.value",
            "description": "BLS periodogram peak power",
        },
        "n_transits": {
            "ucd": "meta.number",
            "description": "Number of observed transits",
        },
        "score": {
            "ucd": "stat.rank",
            "description": "ExoHunter priority score",
        },
        "is_valid": {
            "ucd": "meta.code.qual",
            "description": "Passed all critical validation tests",
        },
    }

    for col_name, meta in _column_metadata.items():
        if col_name in table.colnames:
            col = table[col_name]
            if "ucd" in meta:
                col.meta["ucd"] = meta["ucd"]
            if "unit" in meta:
                col.unit = meta["unit"]
            if "description" in meta:
                col.description = meta["description"]

    # Write as VOTable XML
    table.write(output_path, format="votable", overwrite=True)

    logger.info("Exported %d candidates to %s (VOTable)", len(catalog), output_path)
    return output_path
