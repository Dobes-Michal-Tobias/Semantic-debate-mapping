"""
visualization.py – High-end academic visualisations for the PDT project.

Three figures are produced and saved to ``config.RESULTS_DIR``:
  1. semantic_map.png       – UMAP scatter coloured by party + centroids
  2. entropy_complexity.png – Bubble chart: TTR × avg_sentence_len × aggressiveness
  3. topic_heatmap.png      – Party × topic word-count dominance heatmap
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PARTY_COLORS, RESULTS_DIR, apply_style

logger = logging.getLogger(__name__)

_DPI    = 150
_FIGSIZE_WIDE = (14, 9)
_FIGSIZE_SQ   = (10, 9)


def _party_color(party: str) -> str:
    return PARTY_COLORS.get(party, "#AAAAAA")


# ── 1. Semantic Map ───────────────────────────────────────────────────────────

def plot_semantic_map(
    df: pd.DataFrame,
    coords: np.ndarray,
    centroids_2d: dict[str, np.ndarray] | None = None,
    filename: str = "semantic_map.png",
) -> Path:
    """Scatter of UMAP coordinates coloured by party.

    Parameters
    ----------
    df:
        DataFrame with a ``role_or_party`` column.
    coords:
        (N, 2) array of 2-D projections aligned to *df* rows.
    centroids_2d:
        Optional dict mapping party → 2-D centroid coordinate.  When supplied,
        centroids are drawn as large star markers with party labels.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    parties = df["role_or_party"].unique()

    for party in parties:
        mask = df["role_or_party"] == party
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=_party_color(party),
            label=party,
            alpha=0.45,
            s=18,
            linewidths=0,
        )

    # Party centroids
    if centroids_2d:
        for party, center in centroids_2d.items():
            ax.scatter(
                center[0], center[1],
                c=_party_color(party),
                marker="*",
                s=400,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                party,
                xy=(center[0], center[1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=_party_color(party),
            )

    ax.set_title("Semantic Map of Speech Acts (UMAP projection)", fontsize=15, pad=14)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(
        title="Party / Role",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        framealpha=0.9,
        fontsize=9,
    )
    fig.tight_layout()

    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)
    return out


# ── 2. Entropy × Complexity bubble chart ─────────────────────────────────────

def plot_entropy_complexity(
    df: pd.DataFrame,
    filename: str = "entropy_complexity.png",
) -> Path:
    """Bubble chart: Lexical Diversity (TTR) × Avg Sentence Length,
    bubble size ∝ mean aggressiveness.

    Aggregated at the party level.
    """
    apply_style()

    required = {"ttr", "avg_sentence_len"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    agg: dict = {
        "ttr"             : "mean",
        "avg_sentence_len": "mean",
        "word_count"      : "sum",
    }
    if "aggressiveness" in df.columns and df["aggressiveness"].notna().any():
        agg["aggressiveness"] = "mean"

    party_df = df.groupby("role_or_party").agg(agg).reset_index()
    party_df.rename(columns={"role_or_party": "party"}, inplace=True)

    if "aggressiveness" not in party_df.columns:
        party_df["aggressiveness"] = 1.0

    # Bubble size: scale aggressiveness to point area
    size_scale = 600
    sizes = (party_df["aggressiveness"].fillna(1) + 0.5) * size_scale

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQ)

    colors = [_party_color(p) for p in party_df["party"]]
    sc = ax.scatter(
        party_df["ttr"],
        party_df["avg_sentence_len"],
        s=sizes,
        c=colors,
        alpha=0.80,
        edgecolors="white",
        linewidths=0.8,
    )

    for _, row in party_df.iterrows():
        ax.annotate(
            row["party"],
            xy=(row["ttr"], row["avg_sentence_len"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=9,
            color=_party_color(row["party"]),
            fontweight="bold",
        )

    ax.set_xlabel("Lexical Diversity (mean TTR)")
    ax.set_ylabel("Avg. Sentence Length (words)")
    ax.set_title(
        "Rhetoric Profile: Diversity × Complexity × Aggressiveness",
        fontsize=14, pad=12,
    )

    # Legend for bubble size
    for val, label in [(1, "Low"), (3, "Medium"), (5, "High")]:
        ax.scatter([], [], s=(val + 0.5) * size_scale, c="grey",
                   alpha=0.6, label=f"Aggressiveness: {label}")
    ax.legend(title="Bubble size", framealpha=0.85, fontsize=9)

    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)
    return out


# ── 3. Topic Heatmap ──────────────────────────────────────────────────────────

def plot_topic_heatmap(
    df: pd.DataFrame,
    use_framing: bool = True,
    filename: str = "topic_heatmap.png",
) -> Path:
    """Heatmap of party × topic word-count contribution.

    When *use_framing* is True and ``primary_framing`` column is present,
    framing labels are used as topics; otherwise the ``topic`` column is used.
    """
    apply_style()

    topic_col = (
        "primary_framing"
        if use_framing and "primary_framing" in df.columns and df["primary_framing"].notna().any()
        else "topic"
    )
    logger.info("Topic heatmap using column '%s'.", topic_col)

    pivot = (
        df.groupby(["role_or_party", topic_col])["word_count"]
        .sum()
        .unstack(fill_value=0)
    )

    # Normalise per party (row-normalise → share of words per topic)
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0).round(4)

    fig_height = max(6, len(pivot_norm) * 0.6)
    fig_width  = max(10, len(pivot_norm.columns) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        pivot_norm,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".0%",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Share of total words"},
    )

    ax.set_title(
        f"Topic Dominance Heatmap (Party × {'Framing' if topic_col == 'primary_framing' else 'Topic'})",
        fontsize=14, pad=12,
    )
    ax.set_xlabel(topic_col.replace("_", " ").title())
    ax.set_ylabel("Party / Role")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    fig.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)
    return out


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_all(
    df: pd.DataFrame,
    coords: np.ndarray,
    centroids_2d: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """Generate all three figures and return their paths."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        plot_semantic_map(df, coords, centroids_2d),
        plot_entropy_complexity(df),
        plot_topic_heatmap(df),
    ]
