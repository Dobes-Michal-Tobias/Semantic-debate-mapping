"""
config.py – Global configuration for the Political Discourse Topology project.
All paths are pathlib.Path objects.  Import this module first in every other module.
"""
from pathlib import Path
import seaborn as sns

# ── Directory layout ──────────────────────────────────────────────────────────
# BASE_DIR is the project root (parent of src/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_RAW       : Path = BASE_DIR / "data" / "raw"
DATA_INTERIM   : Path = BASE_DIR / "data" / "interim"
DATA_PROCESSED : Path = BASE_DIR / "data" / "processed"
MODELS_DIR     : Path = BASE_DIR / "models"
RESULTS_DIR    : Path = BASE_DIR / "results"

# Ensure all directories exist on import
for _d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Key file paths ─────────────────────────────────────────────────────────────
TRANSCRIPT_GLOB   : str  = "*.txt"                        # matched inside DATA_RAW
CLEANED_CSV       : Path = DATA_INTERIM   / "cleaned_debates.csv"
EMBEDDINGS_NPY    : Path = DATA_INTERIM   / "embeddings.npy"
METRICS_CSV       : Path = DATA_INTERIM   / "lexical_metrics.csv"
UMAP_CSV          : Path = DATA_INTERIM   / "umap_coords.csv"
ENRICHED_CSV      : Path = DATA_PROCESSED / "final_enriched_data.csv"

# ── Sentence-transformer model ─────────────────────────────────────────────────
EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Czech political-party colour palette ──────────────────────────────────────
# Keys match the role_or_party values produced by the parser
PARTY_COLORS: dict[str, str] = {
    "ANO"       : "#003DA5",   # ANO flagship blue
    "SPOLU"     : "#1B4F9C",   # ODS/SPOLU coalition blue
    "Piráti"    : "#000000",   # Piráti black
    "STAN"      : "#007EC7",   # STAN cyan
    "SPD"       : "#C8102E",   # SPD red
    "Stačilo!"  : "#8B0000",   # Stačilo! dark-red
    "Motoristé" : "#FF6600",   # Motoristé orange
    "Přísaha"   : "#6A0DAD",   # Přísaha purple
    "moderátor" : "#888888",   # neutral grey
}

# ── Seaborn / Matplotlib academic style ───────────────────────────────────────
SEABORN_STYLE: dict = {
    "context"  : "talk",
    "style"    : "whitegrid",
    "palette"  : list(PARTY_COLORS.values()),
    "font_scale": 1.1,
}

def apply_style() -> None:
    """Apply the global seaborn style.  Call once at the start of any plot script."""
    sns.set_theme(
        context=SEABORN_STYLE["context"],
        style=SEABORN_STYLE["style"],
        palette=SEABORN_STYLE["palette"],
        font_scale=SEABORN_STYLE["font_scale"],
    )
