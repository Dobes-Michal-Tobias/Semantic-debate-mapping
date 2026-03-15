# Mapping Semantic Convergence and Lexical Diversity in the 2025 Election Debates

Computational analysis of Czech political debate transcripts — quantitative linguistics,
semantic embeddings, and LLM-powered rhetorical tagging.

> **Paper:** See [`article.pdf`](article.pdf) for the full academic write-up (in Czech).

---

## What it does

The pipeline processes a raw debate transcript and produces three publication-quality figures:

| Figure | Description |
|---|---|
| `semantic_map.png` | UMAP scatter of every speech act, coloured by party, with centroid stars |
| `entropy_complexity.png` | Bubble chart — Lexical Diversity × Avg. Sentence Length × Aggressiveness |
| `topic_heatmap.png` | Party × Topic word-count dominance heatmap |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> `umap-learn` and `anthropic` are optional.
> If `umap-learn` is absent, PCA is used automatically.
> If `anthropic` is absent, the `--enrich` flag cannot be used.

### 2. Place the transcript

Put your UTF-8 encoded `*.txt` debate transcript in `data/raw/`.
Each line must follow the format:

```
Speaker Name (Party) [HH:MM:SS](Topic): Speech text …
```

### 3. Run the pipeline

```bash
# Full pipeline — no LLM call
python main.py

# Include LLM enrichment (sentiment / aggressiveness / framing)
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py --enrich

# Force re-run all stages (ignore cached outputs)
python main.py --force

# Both flags
python main.py --enrich --force
```

Results are written to `results/`.

---

## Project layout

```
FinalTask/
├── main.py                        # Orchestration entry point
├── requirements.txt
├── article.tex                    # LaTeX source of the research paper
├── article.pdf                    # Compiled paper (PDF)
├── TECHNICKA_DOKUMENTACE.md       # Czech technical documentation
├── src/
│   ├── config.py                  # All paths, palette, seaborn style
│   ├── parser.py                  # Transcript → cleaned_debates.csv
│   ├── analytics.py               # TTR/Maas, embeddings, UMAP/PCA, centroids
│   ├── enrichment.py              # Anthropic API batch tagging
│   └── visualization.py           # Three figures
├── data/
│   ├── raw/                       # Source transcripts (*.txt)
│   ├── interim/                   # Intermediate artefacts (auto-generated)
│   └── processed/                 # Enriched CSV (auto-generated)
├── models/                        # Sentence-transformer weights (auto-downloaded)
└── results/                       # Output figures (auto-generated)
```

---

## Pipeline stages

| Stage | Module | Output |
|---|---|---|
| 1. Parse | `parser.py` | `data/interim/cleaned_debates.csv` |
| 2. Lexical metrics | `analytics.py` | `data/interim/lexical_metrics.csv` + columns `ttr`, `maas` |
| 3. Embeddings | `analytics.py` | `data/interim/embeddings.npy` |
| 4. Dim. reduction | `analytics.py` | `data/interim/umap_coords.csv` |
| 5. LLM enrichment | `enrichment.py` | `data/processed/final_enriched_data.csv` |
| 6. Visualisation | `visualization.py` | `results/*.png` |

Every expensive stage is **idempotent**: if the output file already exists with the
correct number of rows, the stage is skipped automatically.

---

## Dataset

- **Source:** Czech Television pre-election debate transcripts (`data/raw/full_transcript.txt`)
- **Period:** September–October 2025
- **Scope:** Topic `1.10 Ekonomika`
- **Size:** 7 715 speech acts
- **Parties:** ANO · SPOLU · Piráti · STAN · SPD · Stačilo! · Motoristé · Přísaha · moderátor

---

## Embedding model

[`paraphrase-multilingual-MiniLM-L12-v2`](https://www.sbert.net/docs/pretrained_models.html)
(sentence-transformers) — 12-layer multilingual MiniLM, 384-dimensional output,
supports Czech out of the box.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Only for `--enrich` | Anthropic API key |

---

## Technical documentation

See [TECHNICKA_DOKUMENTACE.md](TECHNICKA_DOKUMENTACE.md) for a detailed module-by-module
description in Czech.

---

## License

This project was developed as a seminar paper for the course **KRAD — Kritická analýza diskurzu**
at the Department of General Linguistics, Palacký University Olomouc.
