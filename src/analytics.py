"""
analytics.py – Quantitative-linguistics and NLP analytics.

Provides:
  1. Lexical-diversity metrics (TTR, Maas index) per speaker / party
  2. Sentence embeddings via sentence-transformers
  3. Dimensionality reduction (UMAP with PCA fallback)
  4. Per-party centroid vectors
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config import (
    CLEANED_CSV,
    EMBEDDING_MODEL,
    EMBEDDINGS_NPY,
    METRICS_CSV,
    MODELS_DIR,
    UMAP_CSV,
)

logger = logging.getLogger(__name__)


# ── 1. Lexical diversity ──────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace-split tokenisation (fast, language-agnostic)."""
    return text.lower().split()


def ttr(tokens: list[str]) -> float:
    """Type-Token Ratio: |types| / |tokens|.  Returns 0 for empty input."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def maas_index(tokens: list[str]) -> float:
    """Maas (1972) index: (log N – log V) / (log N)².
    Lower values indicate higher lexical richness.
    Returns NaN for < 2 tokens.
    """
    n = len(tokens)
    v = len(set(tokens))
    if n < 2 or v < 1:
        return float("nan")
    log_n = math.log(n)
    log_v = math.log(v)
    denom = log_n ** 2
    if denom == 0:
        return float("nan")
    return (log_n - log_v) / denom


def compute_lexical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute TTR and Maas index per speech act and per party.

    Returns a summary DataFrame with one row per (role_or_party).
    Also attaches per-row columns ``ttr`` and ``maas`` to *df* in place.
    """
    tokens_col = df["content"].map(_tokenize)
    df = df.copy()
    df["ttr"]  = tokens_col.map(ttr)
    df["maas"] = tokens_col.map(maas_index)

    summary = (
        df.groupby("role_or_party")
        .agg(
            total_words   =("word_count", "sum"),
            mean_ttr      =("ttr",        "mean"),
            mean_maas     =("maas",       "mean"),
            speech_count  =("content",    "count"),
        )
        .reset_index()
        .round(4)
    )

    summary.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    logger.info("Lexical metrics saved → %s", METRICS_CSV)
    return summary


# ── 2. Sentence embeddings ────────────────────────────────────────────────────

def compute_embeddings(
    df: pd.DataFrame,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    cache: bool = True,
) -> np.ndarray:
    """Generate sentence embeddings for every speech act in *df*.

    Parameters
    ----------
    cache:
        If True, load from ``EMBEDDINGS_NPY`` when it already exists.

    Returns
    -------
    np.ndarray of shape (N, embedding_dim)
    """
    if cache and EMBEDDINGS_NPY.exists():
        embeddings = np.load(EMBEDDINGS_NPY)
        if embeddings.shape[0] == len(df):
            logger.info("Loaded cached embeddings from %s", EMBEDDINGS_NPY)
            return embeddings
        logger.warning(
            "Cached embeddings shape mismatch (%d vs %d rows) – recomputing.",
            embeddings.shape[0], len(df),
        )

    from sentence_transformers import SentenceTransformer  # lazy import

    logger.info("Loading sentence-transformer model '%s' …", model_name)
    model = SentenceTransformer(model_name, cache_folder=str(MODELS_DIR))

    texts = df["content"].tolist()
    logger.info("Encoding %d speech acts (batch_size=%d) …", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(EMBEDDINGS_NPY, embeddings)
    logger.info("Embeddings saved → %s  shape=%s", EMBEDDINGS_NPY, embeddings.shape)
    return embeddings


# ── 3. Dimensionality reduction ───────────────────────────────────────────────

def reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal["umap", "pca"] = "umap",
    n_components: int = 2,
    cache: bool = True,
    **kwargs,
) -> np.ndarray:
    """Project high-dimensional embeddings into *n_components* dimensions.

    Parameters
    ----------
    method:
        ``"umap"`` (preferred) or ``"pca"`` (fallback when umap not installed).
    kwargs:
        Extra keyword arguments forwarded to the reducer constructor.

    Returns
    -------
    np.ndarray of shape (N, n_components)
    """
    if cache and UMAP_CSV.exists():
        coords = pd.read_csv(UMAP_CSV).values
        if coords.shape[0] == embeddings.shape[0]:
            logger.info("Loaded cached 2D coords from %s", UMAP_CSV)
            return coords
        logger.warning("Cached 2D coords shape mismatch – recomputing.")

    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
            logger.info("Running UMAP …")
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("umap-learn not installed – falling back to PCA.")
            coords = _pca_reduce(embeddings, n_components, **kwargs)
    else:
        coords = _pca_reduce(embeddings, n_components, **kwargs)

    cols = [f"dim_{i}" for i in range(n_components)]
    pd.DataFrame(coords, columns=cols).to_csv(UMAP_CSV, index=False)
    logger.info("2D coords saved → %s", UMAP_CSV)
    return coords


def _pca_reduce(
    embeddings: np.ndarray, n_components: int, **kwargs
) -> np.ndarray:
    from sklearn.decomposition import PCA  # type: ignore

    logger.info("Running PCA …")
    pca = PCA(n_components=n_components, random_state=42, **kwargs)
    return pca.fit_transform(embeddings)


# ── 4. Party centroids ────────────────────────────────────────────────────────

def compute_centroids(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate the mean embedding vector (centroid) for each party.

    Returns
    -------
    dict mapping role_or_party → 1-D centroid array
    """
    centroids: dict[str, np.ndarray] = {}
    for party, group in df.groupby("role_or_party"):
        idx = group.index.tolist()
        centroids[party] = embeddings[idx].mean(axis=0)
        logger.debug("Centroid computed for '%s'  (n=%d)", party, len(idx))

    logger.info("Centroids computed for %d parties/roles.", len(centroids))
    return centroids


def interparty_distance_matrix(
    centroids: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a symmetric DataFrame of cosine distances between party centroids."""
    from sklearn.metrics.pairwise import cosine_distances  # type: ignore

    parties = list(centroids.keys())
    matrix  = np.vstack([centroids[p] for p in parties])
    dist    = cosine_distances(matrix)
    return pd.DataFrame(dist, index=parties, columns=parties).round(4)
