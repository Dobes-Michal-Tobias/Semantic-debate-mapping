"""
parser.py – Structural parsing of Czech political-debate transcripts.

Expected line format (one speech per line, blank lines as separators):
    Name (Party/Role) [HH:MM:SS](Topic): Speech text …

Multi-line continuation lines (lines that do NOT start with a new speaker
header) are appended to the preceding speech entry.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, CLEANED_CSV, TRANSCRIPT_GLOB

logger = logging.getLogger(__name__)

# Matches the full header: "Name (Role) [HH:MM:SS](Topic): content"
_HEADER_RE = re.compile(
    r"^(?P<speaker_name>.+?)"           # speaker name (greedy-safe: stops at last ' (')
    r"\s+\((?P<role_or_party>[^)]+)\)"  # (Party/Role)
    r"\s+\[(?P<timestamp>\d{2}:\d{2}:\d{2})\]"  # [HH:MM:SS]
    r"\((?P<topic>[^)]+)\)"             # (Topic)
    r":\s*(?P<content>.*)"              # : speech text
)


class DebateParser:
    """Parse one or more raw debate transcript files into a tidy DataFrame.

    Parameters
    ----------
    raw_dir:
        Directory that contains the ``*.txt`` transcript files.
        Defaults to ``config.DATA_RAW``.
    glob:
        Glob pattern used to discover files inside *raw_dir*.
    """

    def __init__(
        self,
        raw_dir: Path | None = None,
        glob: str = TRANSCRIPT_GLOB,
    ) -> None:
        self.raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
        self.glob    = glob

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self) -> pd.DataFrame:
        """Parse all transcript files and return a single cleaned DataFrame."""
        files = sorted(self.raw_dir.glob(self.glob))
        if not files:
            raise FileNotFoundError(
                f"No transcript files matching '{self.glob}' found in {self.raw_dir}"
            )

        frames: list[pd.DataFrame] = []
        for path in files:
            logger.info("Parsing %s", path.name)
            frames.append(self._parse_file(path))

        df = pd.concat(frames, ignore_index=True)
        df = self._clean(df)
        logger.info("Parsed %d speech acts from %d file(s)", len(df), len(files))
        return df

    def parse_and_save(self) -> pd.DataFrame:
        """Parse transcripts, save the result to ``CLEANED_CSV``, and return it."""
        df = self.parse()
        CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")
        logger.info("Saved cleaned data → %s", CLEANED_CSV)
        return df

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_file(self, path: Path) -> pd.DataFrame:
        """Parse a single transcript file."""
        text = path.read_text(encoding="utf-8")
        records: list[dict] = []
        current: dict | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue  # skip blank separators

            m = _HEADER_RE.match(line)
            if m:
                if current is not None:
                    records.append(current)
                current = {
                    "source_file"  : path.name,
                    "speaker_name" : m.group("speaker_name").strip(),
                    "role_or_party": m.group("role_or_party").strip(),
                    "timestamp"    : m.group("timestamp").strip(),
                    "topic"        : m.group("topic").strip(),
                    "content"      : m.group("content").strip(),
                }
            elif current is not None:
                # Continuation line: append to current speech
                current["content"] = current["content"] + " " + line

        if current is not None:
            records.append(current)

        return pd.DataFrame(records)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise whitespace and drop empty content rows."""
        df["content"] = df["content"].str.strip()
        df = df[df["content"].str.len() > 0].copy()
        df.reset_index(drop=True, inplace=True)

        # Derived convenience columns
        df["timestamp_seconds"] = df["timestamp"].map(_ts_to_seconds)
        df["word_count"]        = df["content"].str.split().str.len()
        df["sentence_count"]    = df["content"].str.count(r"[.!?]+")
        df["avg_sentence_len"]  = (
            df["word_count"] / df["sentence_count"].replace(0, 1)
        ).round(2)

        return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts_to_seconds(ts: str) -> int:
    """Convert 'HH:MM:SS' to total seconds."""
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s
