"""
enrichment.py – LLM-augmented tagging of debate speech acts.

Uses the Anthropic API (claude-sonnet-4-6) to annotate each speech act with:
  • sentiment      : float in [-1, 1]
  • aggressiveness : int   in [0, 5]
  • primary_framing: str   (e.g. 'Economic threat', 'National sovereignty', …)

Batch processing minimises API calls: multiple speech acts are sent in a single
prompt.  Already-enriched rows are skipped so the module is idempotent.

Environment variable required: ANTHROPIC_API_KEY
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from src.config import CLEANED_CSV, ENRICHED_CSV

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a computational linguistics expert analysing Czech political debate transcripts.
You will receive a JSON array of speech acts.  For EACH item return a JSON array (same
order, same length) where every element has exactly three keys:
  "sentiment"       – float between -1.0 (very negative) and 1.0 (very positive)
  "aggressiveness"  – integer 0 (calm) to 5 (highly aggressive)
  "primary_framing" – short English label for the dominant rhetorical frame
    (e.g. "Economic threat", "National sovereignty", "Social justice",
     "Fiscal responsibility", "EU scepticism", "Party attack", "Neutral/factual")
Return ONLY the JSON array – no prose, no markdown fences."""

_USER_TEMPLATE = "Analyse these {n} speech acts:\n{data}"


def _build_batch_prompt(batch: list[dict]) -> str:
    payload = [
        {"id": row["id"], "speaker": row["speaker"], "text": row["text"]}
        for row in batch
    ]
    return _USER_TEMPLATE.format(n=len(payload), data=json.dumps(payload, ensure_ascii=False))


# ── Main enrichment function ──────────────────────────────────────────────────

def enrich_dataframe(
    df: pd.DataFrame,
    batch_size: int = 20,
    model: str = "claude-sonnet-4-6",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """Attach LLM-generated tags to *df* and save to ``ENRICHED_CSV``.

    Already-enriched rows (those where ``sentiment`` is not NaN) are skipped.

    Parameters
    ----------
    batch_size:
        Number of speech acts per API call.
    model:
        Anthropic model identifier.
    max_retries:
        Number of retries per batch on API errors.
    retry_delay:
        Seconds to wait between retries.

    Returns
    -------
    The enriched DataFrame (also saved to disk).
    """
    import anthropic  # lazy import – not required for other modules

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Export it before running the enrichment step."
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Resume from an existing partial file if available
    if ENRICHED_CSV.exists():
        existing = pd.read_csv(ENRICHED_CSV, encoding="utf-8-sig")
        df = _merge_existing(df, existing)
        logger.info("Resuming enrichment – %d rows already tagged.", df["sentiment"].notna().sum())

    # Rows that still need enrichment
    needs_enrichment = df[df["sentiment"].isna()].copy()
    logger.info("%d speech acts to enrich …", len(needs_enrichment))

    for start in range(0, len(needs_enrichment), batch_size):
        chunk = needs_enrichment.iloc[start : start + batch_size]
        batch_payload = [
            {"id": str(row.name), "speaker": row["speaker_name"], "text": row["content"]}
            for _, row in chunk.iterrows()
        ]
        prompt = _build_batch_prompt(batch_payload)

        results = _call_api(client, model, prompt, max_retries, retry_delay)
        if results is None:
            logger.warning("Batch %d–%d failed – skipping.", start, start + len(chunk))
            continue

        # Map results back by position (same order guaranteed by prompt)
        for i, (orig_idx, _) in enumerate(chunk.iterrows()):
            if i >= len(results):
                break
            r = results[i]
            df.at[orig_idx, "sentiment"]       = float(r.get("sentiment", 0.0))
            df.at[orig_idx, "aggressiveness"]  = int(r.get("aggressiveness", 0))
            df.at[orig_idx, "primary_framing"] = str(r.get("primary_framing", "Unknown"))

        # Checkpoint after every batch
        _save(df)
        logger.info(
            "Enriched rows %d–%d / %d",
            start + 1, min(start + batch_size, len(needs_enrichment)), len(needs_enrichment),
        )

    _save(df)
    logger.info("Enrichment complete.  Saved → %s", ENRICHED_CSV)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_api(
    client,
    model: str,
    prompt: str,
    max_retries: int,
    retry_delay: float,
) -> list[dict] | None:
    """Call the Anthropic API and parse the JSON response."""
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error on attempt %d: %s", attempt + 1, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("API error on attempt %d: %s", attempt + 1, exc)
        time.sleep(retry_delay)
    return None


def _merge_existing(df: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Merge already-computed tags into the current DataFrame."""
    tag_cols = ["sentiment", "aggressiveness", "primary_framing"]
    for col in tag_cols:
        if col not in df.columns:
            df[col] = float("nan") if col == "sentiment" else None
    for col in ["aggressiveness"]:
        if col not in df.columns:
            df[col] = None

    if all(c in existing.columns for c in ["content"] + tag_cols):
        lookup = existing.set_index("content")[tag_cols]
        for col in tag_cols:
            mask = df["content"].isin(lookup.index)
            df.loc[mask, col] = df.loc[mask, "content"].map(lookup[col])
    return df


def _save(df: pd.DataFrame) -> None:
    ENRICHED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENRICHED_CSV, index=False, encoding="utf-8-sig")


def prepare_empty_tag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add NaN-initialised tag columns to a fresh DataFrame."""
    df = df.copy()
    df["sentiment"]       = float("nan")
    df["aggressiveness"]  = pd.NA
    df["primary_framing"] = pd.NA
    return df
