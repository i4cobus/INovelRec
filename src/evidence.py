"""Independent evidence sampling for evaluation.

The recommender builds its profile from three fixed windows — opening (0.0),
middle (0.5), and ending (1.0). If a judge scored relevance from that same text
it would only be answering "does this profile match the query", not "does this
*book* match the query", and would inherit every sampling error the system made.
That is circular, and it would hide exactly the failures evaluation exists to find.

So judge evidence is drawn from *different* offsets in the raw text. Offsets are
derived deterministically from the novel id, so a given book always yields the
same evidence across runs (caching and reproducibility) while different books
sample different parts of their text.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Fractions the Stage 2 profile already uses. Judge windows must avoid these.
PROFILE_FRACTIONS = (0.0, 0.5, 1.0)
MIN_FRACTION_DISTANCE = 0.08

DEFAULT_WINDOWS = 4
DEFAULT_WINDOW_CHARS = 800
WINDOW_SEPARATOR = "\n\n[……]\n\n"


def stable_unit_float(seed: str) -> float:
    """Map a string to a stable float in [0, 1) without touching global RNG state."""

    digest = hashlib.sha1(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def judge_fractions(novel_id: str, windows: int = DEFAULT_WINDOWS) -> list[float]:
    """Pick evenly spread sampling fractions that avoid the profile's windows.

    The text is divided into ``windows`` bands and one offset is chosen inside
    each, jittered by a hash of the novel id. Offsets that land too close to a
    profile fraction are nudged away.
    """

    if windows < 1:
        raise ValueError("windows must be positive")

    fractions: list[float] = []
    for index in range(windows):
        band_start = (index + 0.15) / (windows + 1)
        band_width = 1.0 / (windows + 1)
        jitter = stable_unit_float(f"{novel_id}:{index}") * band_width * 0.7
        fraction = min(max(band_start + jitter, 0.0), 1.0)
        for avoided in PROFILE_FRACTIONS:
            if abs(fraction - avoided) < MIN_FRACTION_DISTANCE:
                fraction = min(avoided + MIN_FRACTION_DISTANCE * 1.5, 0.97)
        fractions.append(round(fraction, 4))
    return fractions


def window_at(text: str, fraction: float, window_chars: int) -> str:
    """Return a window of text starting at a fractional position."""

    if not text or window_chars <= 0:
        return ""
    start = int(len(text) * min(max(fraction, 0.0), 1.0))
    start = min(start, max(len(text) - window_chars, 0))
    return text[start:start + window_chars].strip()


def sample_judge_evidence(
    text: str,
    novel_id: str,
    windows: int = DEFAULT_WINDOWS,
    window_chars: int = DEFAULT_WINDOW_CHARS,
) -> str:
    """Build judge-facing evidence from offsets the system's profile does not use."""

    if not text.strip():
        return ""
    pieces = [window_at(text, fraction, window_chars) for fraction in judge_fractions(novel_id, windows)]
    return WINDOW_SEPARATOR.join(piece for piece in pieces if piece)


def load_raw_text_lookup(inventory_path: Path, novel_ids: set[str]) -> dict[str, str]:
    """Read raw novel text for the requested ids, skipping unreadable files.

    Judge and annotator must read the *same* evidence for their labels to be
    comparable, so both paths load text through here.
    """

    import pandas as pd

    from src.profile import read_text_with_encoding

    inventory = pd.read_parquet(
        inventory_path,
        columns=["novel_id", "absolute_path", "detected_encoding", "read_status", "decode_replacement_chars"],
    )
    inventory = inventory[inventory["novel_id"].astype(str).isin(novel_ids)]
    texts: dict[str, str] = {}
    for row in inventory.to_dict(orient="records"):
        if row.get("read_status") != "ok":
            continue
        try:
            texts[str(row["novel_id"])] = read_text_with_encoding(
                Path(str(row["absolute_path"])),
                row.get("detected_encoding"),
                allow_lossy=int(row.get("decode_replacement_chars", 0) or 0) > 0,
            )
        except (OSError, UnicodeError, LookupError, ValueError):
            continue
    return texts
