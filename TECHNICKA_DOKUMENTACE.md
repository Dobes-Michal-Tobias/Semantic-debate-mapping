# Technická dokumentace — Political Discourse Topology (PDT)

> **Jazyk:** Česky
> **Verze:** 1.0 (2026-03)
> **Předmět:** KRAD — Kvantitativní analýza diskurzu
> **Autor projektu:** viz git log

---

## Obsah

1. [Přehled projektu](#1-přehled-projektu)
2. [Architektura a adresářová struktura](#2-architektura-a-adresářová-struktura)
3. [Datový tok — pipeline](#3-datový-tok--pipeline)
4. [Modul `config.py`](#4-modul-configpy)
5. [Modul `parser.py`](#5-modul-parserpy)
6. [Modul `analytics.py`](#6-modul-analyticspy)
7. [Modul `enrichment.py`](#7-modul-enrichmentpy)
8. [Modul `visualization.py`](#8-modul-visualizationpy)
9. [Orchestrace — `main.py`](#9-orchestrace--mainpy)
10. [Závislosti a instalace](#10-závislosti-a-instalace)
11. [Spuštění a přepínače](#11-spuštění-a-přepínače)
12. [Výstupní artefakty](#12-výstupní-artefakty)
13. [Metodologické poznámky](#13-metodologické-poznámky)

---

## 1. Přehled projektu

**Political Discourse Topology (PDT)** je výzkumná pipeline pro kvantitativní analýzu českých politických debat. Vstupem je surový textový přepis debaty; výstupem jsou tři vědecké vizualizace zachycující sémantické, lexikální a rétorické vlastnosti projevů jednotlivých politických stran.

### Výzkumné otázky

- Jak se liší sémantický prostor projevů jednotlivých stran (shluky ve UMAP projekci)?
- Jaká je lexikální rozmanitost a komplexita vyjadřování napříč stranami (TTR, Maas index)?
- Jaké rétorické rámce strany využívají a s jakou agresivitou?

### Zkoumaná data

| Parametr | Hodnota |
|---|---|
| Zdroj | Přepis politické debaty, téma 1.10 Ekonomika |
| Počet promluv | ~7 715 speech acts |
| Strany / role | ANO, SPOLU, Piráti, STAN, SPD, Stačilo!, Motoristé, Přísaha, moderátor |
| Kódování souboru | UTF-8 |

---

## 2. Architektura a adresářová struktura

```
FinalTask/
├── main.py                        # Vstupní bod, orchestrace 6 fází
├── requirements.txt               # Závislosti
├── src/
│   ├── __init__.py
│   ├── config.py                  # Globální konfigurace (cesty, paleta, styl)
│   ├── parser.py                  # Parsování surového přepisu
│   ├── analytics.py               # Lexikální metriky, embeddingy, redukce dimenzí
│   ├── enrichment.py              # LLM anotace (Anthropic API)
│   └── visualization.py           # Generování grafů
├── data/
│   ├── raw/                       # Zdrojové *.txt přepisy (vstup)
│   ├── interim/                   # Mezivýsledky (automaticky generováno)
│   │   ├── cleaned_debates.csv
│   │   ├── lexical_metrics.csv
│   │   ├── embeddings.npy
│   │   └── umap_coords.csv
│   └── processed/                 # Obohacená data (po --enrich)
│       └── final_enriched_data.csv
├── models/                        # Cache sentence-transformer modelu
└── results/                       # Výstupní obrázky PNG
    ├── semantic_map.png
    ├── entropy_complexity.png
    └── topic_heatmap.png
```

### Datová vrstva (princip idempotence)

Každá fáze kontroluje existenci výstupního souboru a shodu počtu řádků
s aktuálním datasetem. Pokud soubor existuje a počty sedí, fáze se přeskočí.
Přepínač `--force` toto chování potlačí a vynutí přepočítání.

---

## 3. Datový tok — pipeline

```
data/raw/*.txt
      │
      ▼ Fáze 1 – Parsování (parser.py)
cleaned_debates.csv  (7 715 řádků, 9 sloupců)
      │
      ▼ Fáze 2 – Lexikální metriky (analytics.py)
lexical_metrics.csv  +  sloupce ttr, maas v df
      │
      ▼ Fáze 3 – Sentence embeddingy (analytics.py)
embeddings.npy  (7 715 × 384)
      │
      ▼ Fáze 4 – Redukce dimenzí UMAP/PCA (analytics.py)
umap_coords.csv  (7 715 × 2)
      │
      ▼ Fáze 5 – LLM anotace [volitelné] (enrichment.py)
final_enriched_data.csv  (+sloupce sentiment, aggressiveness, primary_framing)
      │
      ▼ Fáze 6 – Vizualizace (visualization.py)
results/semantic_map.png
results/entropy_complexity.png
results/topic_heatmap.png
```

---

## 4. Modul `config.py`

**Účel:** Jediné místo pro všechny cesty a globální nastavení. Každý jiný modul
importuje pouze z tohoto souboru — žádné hardcoded cesty.

### Klíčové konstanty

| Konstanta | Typ | Popis |
|---|---|---|
| `BASE_DIR` | `Path` | Kořen projektu (rodič složky `src/`) |
| `DATA_RAW` | `Path` | `BASE_DIR/data/raw` |
| `DATA_INTERIM` | `Path` | `BASE_DIR/data/interim` |
| `DATA_PROCESSED` | `Path` | `BASE_DIR/data/processed` |
| `MODELS_DIR` | `Path` | `BASE_DIR/models` |
| `RESULTS_DIR` | `Path` | `BASE_DIR/results` |
| `CLEANED_CSV` | `Path` | Výstup parsování |
| `EMBEDDINGS_NPY` | `Path` | Binární cache embeddingů |
| `METRICS_CSV` | `Path` | Lexikální metriky za stranu |
| `UMAP_CSV` | `Path` | 2D souřadnice po redukci dimenzí |
| `ENRICHED_CSV` | `Path` | Obohacená data (LLM tagy) |
| `EMBEDDING_MODEL` | `str` | `"paraphrase-multilingual-MiniLM-L12-v2"` |
| `PARTY_COLORS` | `dict` | Hex barvy stran (9 položek) |

### Paleta stran

```python
PARTY_COLORS = {
    "ANO"       : "#003DA5",   # modrá
    "SPOLU"     : "#1B4F9C",   # koaliční modrá
    "Piráti"    : "#000000",   # černá
    "STAN"      : "#007EC7",   # azurová
    "SPD"       : "#C8102E",   # červená
    "Stačilo!"  : "#8B0000",   # tmavě červená
    "Motoristé" : "#FF6600",   # oranžová
    "Přísaha"   : "#6A0DAD",   # fialová
    "moderátor" : "#888888",   # šedá
}
```

Klíče odpovídají přesně hodnotám sloupce `role_or_party` generovaným parserem.

### Funkce `apply_style()`

Nastaví globální Seaborn téma (`whitegrid`, kontext `talk`, `font_scale=1.1`).
Volá se na začátku každé vykreslovací funkce.

---

## 5. Modul `parser.py`

**Účel:** Transformace surového textového přepisu na strukturovaný tidy DataFrame.

### Formát vstupního souboru

Každá promluva je na jednom řádku (nebo na více řádcích, pokud text pokračuje na dalším řádku bez nové hlavičky):

```
Jméno Příjmení (Strana) [HH:MM:SS](Téma): Text promluvy …
```

Prázdné řádky jsou přeskočeny. Pokračovací řádky (nezačínají hlavičkou) jsou připojeny k předchozí promluvě.

### Regulární výraz hlavičky

```python
_HEADER_RE = re.compile(
    r"^(?P<speaker_name>.+?)"
    r"\s+\((?P<role_or_party>[^)]+)\)"
    r"\s+\[(?P<timestamp>\d{2}:\d{2}:\d{2})\]"
    r"\((?P<topic>[^)]+)\)"
    r":\s*(?P<content>.*)"
)
```

Skupiny: `speaker_name`, `role_or_party`, `timestamp`, `topic`, `content`.

### Třída `DebateParser`

```python
parser = DebateParser()        # hledá v DATA_RAW
df = parser.parse_and_save()   # parsuje a uloží cleaned_debates.csv
```

Metoda `_clean()` přidává odvozené sloupce:

| Sloupec | Výpočet |
|---|---|
| `timestamp_seconds` | Převod `HH:MM:SS` na celé sekundy |
| `word_count` | Počet mezer + 1 |
| `sentence_count` | Počet výskytů `[.!?]+` |
| `avg_sentence_len` | `word_count / max(sentence_count, 1)` |

### Výstupní schéma `cleaned_debates.csv`

| Sloupec | Typ | Popis |
|---|---|---|
| `source_file` | str | Název zdrojového souboru |
| `speaker_name` | str | Celé jméno řečníka |
| `role_or_party` | str | Strana nebo role (klíč do `PARTY_COLORS`) |
| `timestamp` | str | `HH:MM:SS` |
| `topic` | str | Téma (z přepisu) |
| `content` | str | Plný text promluvy |
| `timestamp_seconds` | int | Čas v sekundách |
| `word_count` | int | Počet slov |
| `sentence_count` | int | Počet vět |
| `avg_sentence_len` | float | Průměrná délka věty ve slovech |

---

## 6. Modul `analytics.py`

**Účel:** Kvantitativní lingvistika a NLP — čtyři analytické vrstvy.

### 6.1 Lexikální diverzita

#### TTR (Type-Token Ratio)

```
TTR = |V| / |N|
```

kde `|V|` = počet unikátních typů, `|N|` = celkový počet tokenů.
Hodnota 1,0 = každé slovo je unikátní; hodnota blízká 0 = velká opakovanost.

#### Maas index (1972)

```
Maas = (log N − log V) / (log N)²
```

Nižší hodnota = vyšší lexikální bohatství. Vhodný pro texty různé délky —
méně citlivý na délku textu než TTR.

Tokenizace je záměrně jednoduchá (lowercase, split na mezerách) — jazykově
agnostická, dostatečná pro komparativní analýzu.

#### Funkce `compute_lexical_metrics(df)`

Vrací souhrnný DataFrame za stranu a ukládá `lexical_metrics.csv`:

| Sloupec | Popis |
|---|---|
| `role_or_party` | Strana |
| `total_words` | Celkový počet slov |
| `mean_ttr` | Průměrné TTR promluv |
| `mean_maas` | Průměrný Maas index |
| `speech_count` | Počet promluv |

### 6.2 Sentence embeddingy

Model: **`paraphrase-multilingual-MiniLM-L12-v2`** (sentence-transformers)

- 12vrstvý multilinguální MiniLM
- Výstupní dimenze: 384
- Nativní podpora češtiny

```python
embeddings = compute_embeddings(df, batch_size=64, cache=True)
# → np.ndarray tvaru (N, 384), uloženo do embeddings.npy
```

Model se stahuje automaticky do `MODELS_DIR` při prvním spuštění.

### 6.3 Redukce dimenzí

Primární metoda: **UMAP** (`umap-learn`)
Záložní metoda: **PCA** (`sklearn`) — automaticky použito, pokud `umap-learn` není nainstalován.

```python
coords = reduce_dimensions(embeddings, method="umap", n_components=2)
# → np.ndarray tvaru (N, 2), uloženo do umap_coords.csv
```

UMAP parametry: `n_components=2`, `random_state=42` (reprodukovatelnost).

### 6.4 Centroidy stran

```python
centroids = compute_centroids(df, embeddings)
# → dict: strana → np.ndarray tvaru (384,)
```

Centroid = průměrný embedding vektor všech promluv dané strany v plném
384D prostoru.

#### Matice inter-stranových vzdáleností

```python
dist_matrix = interparty_distance_matrix(centroids)
# → pd.DataFrame (9×9) cosine vzdáleností
```

Cosine vzdálenost (1 − cosine similarity) měří orientační odlišnost
rétorických pozic stran v sémantickém prostoru.

---

## 7. Modul `enrichment.py`

**Účel:** Obohacení každé promluvy o LLM-generované tagy pomocí Anthropic API.

### Požadavky

- Nastavená proměnná prostředí `ANTHROPIC_API_KEY`
- Nainstalovaný balíček `anthropic`

### Generované anotace

| Tag | Typ | Rozsah | Popis |
|---|---|---|---|
| `sentiment` | `float` | −1,0 až 1,0 | Polarita projevu |
| `aggressiveness` | `int` | 0 až 5 | Míra agresivity vyjadřování |
| `primary_framing` | `str` | viz níže | Dominantní rétorický rámec |

#### Používané rétorické rámce (`primary_framing`)

- `Economic threat` — ekonomická hrozba
- `National sovereignty` — národní suverenita
- `Social justice` — sociální spravedlnost
- `Fiscal responsibility` — fiskální zodpovědnost
- `EU scepticism` — euroskepticismus
- `Party attack` — útok na politického soupeře
- `Neutral/factual` — neutrální / faktický výrok

### Dávkové zpracování

Promluvy jsou odesílány v dávkách (výchozí `batch_size=20`) v jednom API volání.
Systémový prompt instruuje model vrátit JSON pole stejné délky jako vstup.

```
System prompt → role experta na komputační lingvistiku
User prompt   → JSON pole speech actů s poli: id, speaker, text
Response      → JSON pole: [{sentiment, aggressiveness, primary_framing}, …]
```

### Idempotence a checkpointing

- Již otagované řádky (kde `sentiment` není NaN) jsou přeskočeny.
- Po každé dávce se výsledek okamžitě uloží na disk (`ENRICHED_CSV`).
- Pokud je spuštění přerušeno, při dalším spuštění pokračuje od posledního checkpointu.

### Ošetření chyb

- Automatický retry (výchozí `max_retries=3`, `retry_delay=5 s`).
- Případné markdown fence v odpovědi LLM je ořezáno před JSON parsováním.
- Při selhání celé dávky je varování zalogováno a pipeline pokračuje.

---

## 8. Modul `visualization.py`

**Účel:** Generování tří; vizualizací ve formátu PNG (150 DPI).

### Obrázek 1 — `semantic_map.png`

**Typ:** Scatter plot (UMAP projekce)

Každý bod = jedna promluva, barva = strana. Hvězdičky označují 2D centroidy stran.
Centroid je průměr 2D souřadnic promluv dané strany (nikoli projekce centroidu z 384D prostoru).

```python
plot_semantic_map(df, coords, centroids_2d)
```

- Průhlednost bodů: `alpha=0.45`, velikost: `s=18`
- Centroid hvězda: `s=400`, bílý obrys
- Legenda: mimo oblast grafu vpravo

### Obrázek 2 — `entropy_complexity.png`

**Typ:** Bubble chart

- Osa X: průměrné TTR strany (lexikální diverzita)
- Osa Y: průměrná délka věty ve slovech (komplexita)
- Velikost bubliny: průměrná agresivita (pokud jsou LLM tagy dostupné, jinak konstanta 1)

```python
plot_entropy_complexity(df)
```

Každá strana je zastoupena jednou bubblinou. Graf umožňuje srovnat rétorický
styl stran ve třech dimenzích najednou.

### Obrázek 3 — `topic_heatmap.png`

**Typ:** Heatmap (seaborn)

Řádky = strany, sloupce = témata nebo rétorické rámce.
Hodnoty = podíl slov strany věnovaných danému tématu (řádková normalizace).

```python
plot_topic_heatmap(df, use_framing=True)
```

Pokud jsou dostupné LLM tagy (`primary_framing`), použijí se jako sloupce.
Jinak se použije originální sloupec `topic` z přepisu.

Barevná škála `YlOrRd` — vyšší hodnota = větší důraz strany na dané téma.

---

## 9. Orchestrace — `main.py`

### Vstupní bod a CLI

```bash
python main.py [--enrich] [--force]
```

| Přepínač | Efekt |
|---|---|
| _(žádný)_ | Spustí fáze 1–4 a 6 (bez LLM) |
| `--enrich` | Přidá fázi 5 (LLM anotace) |
| `--force` | Ignoruje všechny cache soubory, přepočítá vše |

### Logování

Formát: `HH:MM:SS  LEVEL     modul – zpráva`

Každá fáze loguje stav `[SKIP]` (přeskočeno) nebo `[RUN ]` (spouštění).

### Pomocné funkce

#### `_load_or_skip(path, n_rows)`

Načte CSV, pokud existuje a má správný počet řádků. Jinak vrátí `None`.

#### `_npy_matches(path, n_rows)`

Zkontroluje, zda binární soubor `.npy` existuje a `shape[0]` odpovídá `n_rows`.

### Tok dat mezi fázemi

```python
df        = stage_parse(force)        # pd.DataFrame
df        = stage_lexical(df, force)  # přidá sloupce ttr, maas
embeddings= stage_embeddings(df, force) # np.ndarray (N, 384)
coords    = stage_reduce(embeddings, len(df), force) # np.ndarray (N, 2)
df        = stage_enrich(df, force)   # pouze při --enrich
stage_visualise(df, coords, embeddings, force)
```

---

## 10. Závislosti a instalace

### Instalace

```bash
pip install -r requirements.txt
```

### Přehled závislostí

| Balíček | Verze | Použití |
|---|---|---|
| `numpy` | ≥ 1.26 | Numerické operace, binární cache embeddingů |
| `pandas` | ≥ 2.1 | Tabulková data, CSV I/O |
| `scikit-learn` | ≥ 1.4 | PCA, cosine distances |
| `matplotlib` | ≥ 3.8 | Vykreslování grafů |
| `seaborn` | ≥ 0.13 | Heatmap, globální styl |
| `sentence-transformers` | ≥ 2.7 | Multilinguální embeddingy |
| `umap-learn` | ≥ 0.5 | UMAP redukce dimenzí *(volitelné)* |
| `anthropic` | ≥ 0.25 | Anthropic API client *(volitelné, jen `--enrich`)* |

> **Poznámka k `umap-learn`:** Pokud není nainstalován, pipeline automaticky
> přepne na PCA. Výsledné 2D souřadnice budou lineární projekcí místo
> nelineárního UMAP.

---

## 11. Spuštění a přepínače

### Základní spuštění (bez LLM)

```bash
python main.py
```

Vyprodukuje všechny tři grafy. Fáze 5 (LLM) je přeskočena.

### Spuštění s LLM anotací

```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python main.py --enrich

# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py --enrich
```

### Přepočítání od začátku

```bash
python main.py --force
python main.py --enrich --force
```

### Typický výstup logu

```
10:42:01  INFO      pdt.main – ═══ Political Discourse Topology Pipeline ═══
10:42:01  INFO      pdt.main – BASE_DIR : /path/to/FinalTask
10:42:01  INFO      pdt.main – force    : False  |  enrich: False
10:42:01  INFO      pdt.main – [SKIP] Parsing – loading existing cleaned_debates.csv
10:42:01  INFO      pdt.main – Dataset : 7715 speech acts, 9 parties/roles
10:42:01  INFO      pdt.main – [SKIP] Lexical metrics – using cached lexical_metrics.csv
10:42:03  INFO      pdt.main – [SKIP] Embeddings – loading cached embeddings.npy
10:42:03  INFO      pdt.main – [SKIP] Dim-reduction – using cached umap_coords.csv
10:42:03  INFO      pdt.main – [SKIP] LLM enrichment (pass --enrich to enable).
10:42:03  INFO      pdt.main – [RUN ] Generating visualisations …
10:42:08  INFO      pdt.main – ═══ Pipeline complete.  Results in: results ═══
```

---

## 12. Výstupní artefakty

### Mezivýsledky (`data/interim/`)

| Soubor | Formát | Popis |
|---|---|---|
| `cleaned_debates.csv` | CSV UTF-8-sig | Vyčištěné promluvy, 9 sloupců |
| `lexical_metrics.csv` | CSV UTF-8-sig | Agregované metriky za stranu |
| `embeddings.npy` | NumPy binary | Tensor shape `(N, 384)` |
| `umap_coords.csv` | CSV | 2D souřadnice, sloupce `dim_0`, `dim_1` |

### Obohacená data (`data/processed/`)

| Soubor | Formát | Popis |
|---|---|---|
| `final_enriched_data.csv` | CSV UTF-8-sig | Vše z cleaned + sentiment, aggressiveness, primary_framing |

### Výsledky (`results/`)

| Soubor | Rozlišení | Popis |
|---|---|---|
| `semantic_map.png` | 150 DPI | UMAP scatter stran |
| `entropy_complexity.png` | 150 DPI | Bubble chart rétorického profilu |
| `topic_heatmap.png` | 150 DPI | Heatmap dominance témat/rámců |

---

## 13. Metodologické poznámky

### Volba embeddingového modelu

`paraphrase-multilingual-MiniLM-L12-v2` byl zvolen jako kompromis mezi:
- **Jazykovou pokrytostí:** nativní podpora češtiny (trénován na 50+ jazycích)
- **Kvalitou reprezentace:** parafráze-optimalizovaný model zachycuje sémantiku lépe než generické modely
- **Výpočetní náročností:** 384D výstup je výrazně menší než 768D/1024D modely; zpracování ~7 700 promluv trvá minuty na CPU

### UMAP vs. PCA

UMAP (Uniform Manifold Approximation and Projection) zachovává lokální i globální strukturu dat lépe než lineární PCA. Shlukování podobných projevů ve 2D prostoru je proto věrohodněji interpretovatelné. PCA je k dispozici jako záložní metoda pro prostředí bez `umap-learn`.

### Lexikální metriky

**TTR** je jednoduchý, ale silně závislý na délce textu (delší texty mají přirozeně nižší TTR). **Maas index** tuto závislost tlumí logaritmickou transformací, proto je vhodnější pro komparativní analýzu promluv různých délek.

### LLM anotace

Anotace jsou generovány modelem `claude-sonnet-4-6` v angličtině (rétorické rámce jsou záměrně v angličtině pro konzistenci napříč případnými budoucími srovnávacími studiemi). Batch přístup 20 promluv/volání balancuje mezi latencí API, spolehlivostí parsování odpovědi a náklady.

### Reprodukovatelnost

- UMAP a PCA používají `random_state=42`
- Embeddingy jsou cachovaný jako `.npy` — opakovaný běh bez `--force` vrátí identické výsledky
- LLM anotace jsou nedeterministické (teplota není nastavena na 0) — pro plnou reprodukovatelnost je třeba uložit a verzovat `final_enriched_data.csv`
