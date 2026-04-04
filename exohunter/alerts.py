"""Automatic alert system for new transit candidate detections.

Generates alerts when the pipeline discovers candidates classified as
``NEW_CANDIDATE`` (not in any catalog) that pass validation.  Alerts
are dispatched to two channels:

    1. **File** — a JSON file per batch run in ``data/alerts/``, containing
       all alert-worthy candidates with their parameters.
    2. **Webhook** — an HTTP POST to a configured URL (Slack, Discord,
       Telegram, or any service accepting JSON).  Disabled by default.

Alert trigger criteria:
    - ``xmatch_class == "NEW_CANDIDATE"``
    - ``snr >= MIN_SNR`` (default 7.0)
    - ``is_valid == True``
    - Optionally: ``ml_class == "planet"`` (if ML classifier was run)

Usage::

    from exohunter.alerts import check_and_dispatch_alerts

    # At the end of a batch run:
    check_and_dispatch_alerts(summary_df, sector=56)

Configuration in ``config.py``::

    ALERTS_WEBHOOK_URL = "https://hooks.slack.com/services/..."
"""

from __future__ import annotations

import json
import time as time_module
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from exohunter import config
from exohunter.utils.logging import get_logger

logger = get_logger(__name__)


def find_alertable_candidates(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a batch summary DataFrame to only alert-worthy candidates.

    Criteria:
        - ``xmatch_class == "NEW_CANDIDATE"``
        - ``snr >= config.MIN_SNR``
        - ``is_valid == True``

    Args:
        summary_df: The summary DataFrame from ``run_batch()``.

    Returns:
        A filtered DataFrame containing only alert-worthy rows.
    """
    if summary_df.empty:
        return summary_df

    mask = pd.Series(True, index=summary_df.index)

    if "xmatch_class" in summary_df.columns:
        mask &= summary_df["xmatch_class"] == "NEW_CANDIDATE"

    if "snr" in summary_df.columns:
        snr = pd.to_numeric(summary_df["snr"], errors="coerce")
        mask &= snr >= config.MIN_SNR

    if "is_valid" in summary_df.columns:
        mask &= summary_df["is_valid"] == True

    return summary_df[mask].copy()


def _build_alert_payload(
    alertable: pd.DataFrame,
    sector: int | None = None,
) -> dict:
    """Build a structured alert payload from alert-worthy candidates.

    Args:
        alertable: Filtered DataFrame of alert-worthy candidates.
        sector: TESS sector number (for labelling).

    Returns:
        A JSON-serializable dict with alert metadata and candidate list.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    candidates_list = []
    for _, row in alertable.iterrows():
        entry = {
            "tic_id": str(row.get("tic_id", "")),
            "period": float(row["period"]) if pd.notna(row.get("period")) else None,
            "depth_pct": float(row["depth"]) * 100 if pd.notna(row.get("depth")) else None,
            "snr": float(row["snr"]) if pd.notna(row.get("snr")) else None,
            "xmatch_class": str(row.get("xmatch_class", "")),
        }
        if "ml_class" in row and pd.notna(row.get("ml_class")) and row["ml_class"]:
            entry["ml_class"] = str(row["ml_class"])
        if "ml_prob_planet" in row and pd.notna(row.get("ml_prob_planet")):
            entry["ml_prob_planet"] = float(row["ml_prob_planet"])
        if "name" in row and pd.notna(row.get("name")) and row["name"]:
            entry["name"] = str(row["name"])
        candidates_list.append(entry)

    return {
        "alert_type": "new_candidates",
        "timestamp": timestamp,
        "sector": sector,
        "n_candidates": len(candidates_list),
        "candidates": candidates_list,
    }


def save_alert_file(
    payload: dict,
    sector: int | None = None,
) -> Path:
    """Write an alert payload to a JSON file in ``data/alerts/``.

    Args:
        payload: The alert payload dict.
        sector: TESS sector number (used in the filename).

    Returns:
        Path to the written JSON file.
    """
    alerts_dir = config.ALERTS_DIR
    alerts_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sector_str = f"sector_{sector:02d}" if sector is not None else "manual"
    filename = f"alert_{sector_str}_{timestamp_str}.json"

    path = alerts_dir / filename

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Alert file saved: %s (%d candidates)", path, payload["n_candidates"])
    return path


def send_webhook(payload: dict, url: str | None = None) -> bool:
    """Send an alert payload to a webhook URL via HTTP POST.

    The payload is formatted as a JSON body.  For Slack-compatible
    webhooks, the message is wrapped in a ``text`` field with a
    human-readable summary.

    Args:
        payload: The alert payload dict.
        url: Webhook URL.  If ``None``, uses ``config.ALERTS_WEBHOOK_URL``.

    Returns:
        ``True`` if the POST succeeded, ``False`` otherwise.
    """
    if url is None:
        url = config.ALERTS_WEBHOOK_URL

    if not url:
        logger.debug("No webhook URL configured — skipping")
        return False

    # Build a human-readable summary for Slack/Discord
    n = payload["n_candidates"]
    sector = payload.get("sector", "?")
    lines = [f"*ExoHunter Alert* — {n} new candidate(s) in sector {sector}"]

    for c in payload["candidates"][:5]:  # show top 5
        tic = c.get("tic_id", "?")
        period = c.get("period")
        snr = c.get("snr")
        p_str = f"P={period:.4f}d" if period else "P=?"
        s_str = f"SNR={snr:.1f}" if snr else ""
        ml = c.get("ml_class", "")
        ml_str = f" [ML: {ml}]" if ml else ""
        lines.append(f"  {tic}: {p_str} {s_str}{ml_str}")

    if n > 5:
        lines.append(f"  ... and {n - 5} more")

    message = "\n".join(lines)

    # POST as JSON
    body = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Webhook alert sent (%d): %s", resp.status, url[:50])
            return True
    except Exception:
        logger.warning("Webhook alert failed", exc_info=True)
        return False


def check_and_dispatch_alerts(
    summary_df: pd.DataFrame,
    sector: int | None = None,
) -> int:
    """Check for alert-worthy candidates and dispatch alerts.

    This is the main entry point — call it at the end of a batch run.

    1. Filters ``summary_df`` for NEW_CANDIDATE with SNR >= 7 and valid.
    2. If any found, saves a JSON alert file to ``data/alerts/``.
    3. If a webhook URL is configured, sends an HTTP POST.

    Args:
        summary_df: The summary DataFrame from ``run_batch()``.
        sector: TESS sector number.

    Returns:
        The number of alert-worthy candidates found.
    """
    alertable = find_alertable_candidates(summary_df)

    if alertable.empty:
        logger.info("No alert-worthy candidates found")
        return 0

    logger.info(
        "Found %d alert-worthy NEW_CANDIDATE(s) — dispatching alerts",
        len(alertable),
    )

    payload = _build_alert_payload(alertable, sector=sector)

    # Channel 1: file
    save_alert_file(payload, sector=sector)

    # Channel 2: webhook (if configured)
    send_webhook(payload)

    return len(alertable)
