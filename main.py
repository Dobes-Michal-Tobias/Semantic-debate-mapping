"""
main.py – Orchestration entry point for the Political Discourse Topology pipeline.

Pipeline stages:
  1. Parse raw transcripts → data/interim/cleaned_debates.csv
  2. Compute lexical metrics (TTR, Maas)
  3. Compute sentence embeddings → data/interim/embeddings.npy
  4. Reduce dimensions (UMAP/PCA) → data/interim/umap_coords.csv
  5. (Optional) LLM enrichment → data/processed/final_enriched_data.csv
  6. Generate visualisations → results/

Each expensive stage is guarded by an idempotency check: if the output file
already exists and its row count matches the current dataset, the stage is
skipped.

Usage
-----
  python main.py                  # full pipeline (skip LLM enrichment)
  python main.py --enrich         # include LLM enrichment (requires ANTHROPIC_API_KEY)
  python main.py --force          # re-run all stages (ignore cached outputs)
  python main.py --enrich --force # force-re-run everything including LLM
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging setup (before any local imports that call logging) ────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pdt.main")

# ── Local imports ─────────────────────────────────────────────────────────────
from src import config
from src.config import (
    CLEANED_CSV,
    EMBEDDINGS_NPY,
    ENRICHED_CSV,
    METRICS_CSV,
    UMAP_CSV,
)
from src.parser        import DebateParser
from src.analytics     import (
    compute_embeddings,
    compute_lexical_metrics,
    compute_centroids,
    reduce_dimensions,
    interparty_distance_matrix,
)
from src.enrichment    import enrich_dataframe, prepare_empty_tag_columns
from src.visualization import generate_all


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_or_skip(path: Path, n_rows: int | None = None) -> pd.DataFrame | None:
    """Return a DataFrame if *path* exists (and optionally row-count matches)."""
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if n_rows is not None and len(df) != n_rows:
        logger.warning(
            "Cached file %s has %d rows, expected %d – will recompute.",
            path.name, len(df), n_rows,
        )
        return None
    return df


def _npy_matches(path: Path, n_rows: int) -> bool:
    if not path.exists():
        return False
    arr = np.load(path)
    if arr.shape[0] != n_rows:
        logger.warning(
            "Cached %s has %d rows, expected %d – will recompute.",
            path.name, arr.shape[0], n_rows,
        )
        return False
    return True


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_parse(force: bool) -> pd.DataFrame:
    if not force and CLEANED_CSV.exists():
        logger.info("[SKIP] Parsing – loading existing %s", CLEANED_CSV.name)
        return pd.read_csv(CLEANED_CSV, encoding="utf-8-sig")

    logger.info("[RUN ] Parsing raw transcripts …")
    parser = DebateParser()
    return parser.parse_and_save()


def stage_lexical(df: pd.DataFrame, force: bool) -> pd.DataFrame:
    cached = None if force else _load_or_skip(METRICS_CSV)
    if cached is not None:
        logger.info("[SKIP] Lexical metrics – using cached %s", METRICS_CSV.name)
        # Attach per-row TTR / Maas to df (recompute lightweight)
        from src.analytics import ttr, maas_index, _tokenize
        tokens = df["content"].map(_tokenize)
        df = df.copy()
        df["ttr"]  = tokens.map(ttr)
        df["maas"] = tokens.map(maas_index)
        return df

    logger.info("[RUN ] Computing lexical metrics …")
    compute_lexical_metrics(df)          # saves METRICS_CSV
    # Attach per-row columns
    from src.analytics import ttr, maas_index, _tokenize
    tokens = df["content"].map(_tokenize)
    df = df.copy()
    df["ttr"]  = tokens.map(ttr)
    df["maas"] = tokens.map(maas_index)
    return df


def stage_embeddings(df: pd.DataFrame, force: bool) -> np.ndarray:
    if not force and _npy_matches(EMBEDDINGS_NPY, len(df)):
        logger.info("[SKIP] Embeddings – loading cached %s", EMBEDDINGS_NPY.name)
        return np.load(EMBEDDINGS_NPY)

    logger.info("[RUN ] Computing sentence embeddings …")
    return compute_embeddings(df, cache=not force)


def stage_reduce(embeddings: np.ndarray, n_rows: int, force: bool) -> np.ndarray:
    cached_df = None if force else _load_or_skip(UMAP_CSV, n_rows)
    if cached_df is not None:
        logger.info("[SKIP] Dim-reduction – using cached %s", UMAP_CSV.name)
        return cached_df.values

    logger.info("[RUN ] Running dimensionality reduction …")
    return reduce_dimensions(embeddings, cache=not force)


def stage_enrich(df: pd.DataFrame, force: bool) -> pd.DataFrame:
    cached = None if force else _load_or_skip(ENRICHED_CSV, len(df))
    if cached is not None and cached["sentiment"].notna().all():
        logger.info("[SKIP] Enrichment – all rows already tagged (%s).", ENRICHED_CSV.name)
        return cached

    logger.info("[RUN ] Running LLM enrichment …")
    if "sentiment" not in df.columns:
        df = prepare_empty_tag_columns(df)
    return enrich_dataframe(df)


def stage_visualise(
    df: pd.DataFrame,
    coords: np.ndarray,
    embeddings: np.ndarray,
    force: bool,
) -> None:
    out_files = [
        config.RESULTS_DIR / "semantic_map.png",
        config.RESULTS_DIR / "entropy_complexity.png",
        config.RESULTS_DIR / "topic_heatmap.png",
    ]
    if not force and all(f.exists() for f in out_files):
        logger.info("[SKIP] Visualisations already exist in %s/", config.RESULTS_DIR.name)
        return

    logger.info("[RUN ] Generating visualisations …")
    # Compute 2-D centroids from the already-projected coords
    centroids_high = compute_centroids(df, embeddings)
    centroids_2d: dict[str, np.ndarray] = {}
    for party, group in df.groupby("role_or_party"):
        idx = group.index.tolist()
        centroids_2d[party] = coords[idx].mean(axis=0)

    generated = generate_all(df, coords, centroids_2d)
    for path in generated:
        logger.info("  → %s", path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Political Discourse Topology – analysis pipeline"
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Run the optional LLM enrichment stage (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore all cached outputs and re-run every stage.",
    )
    args = parser.parse_args()

    logger.info("═══ Political Discourse Topology Pipeline ═══")
    logger.info("BASE_DIR : %s", config.BASE_DIR)
    logger.info("force    : %s  |  enrich: %s", args.force, args.enrich)

    # Stage 1 – Parse
    df = stage_parse(args.force)
    logger.info("Dataset : %d speech acts, %d parties/roles",
                len(df), df["role_or_party"].nunique())

    # Stage 2 – Lexical metrics
    df = stage_lexical(df, args.force)

    # Stage 3 – Embeddings
    embeddings = stage_embeddings(df, args.force)

    # Stage 4 – Dimensionality reduction
    coords = stage_reduce(embeddings, len(df), args.force)

    # Stage 5 – (Optional) LLM enrichment
    if args.enrich:
        df = stage_enrich(df, args.force)
    else:
        logger.info("[SKIP] LLM enrichment (pass --enrich to enable).")
        if ENRICHED_CSV.exists():
            cached = _load_or_skip(ENRICHED_CSV, len(df))
            if cached is not None:
                df = cached    # use enriched version if already computed

    # Stage 6 – Visualisations
    stage_visualise(df, coords, embeddings, args.force)

    logger.info("═══ Pipeline complete.  Results in: %s ═══", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
