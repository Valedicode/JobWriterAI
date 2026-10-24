"""Reading teacher label files. Kept free of the app, so analyses of finished runs need no API key."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ("low", "medium", "high", "excellent")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def latest(records: list[dict]) -> dict[int, dict]:
    """The last record of each row (a row that failed and was retried has several)."""
    return {r["row_id"]: r for r in records}


def spearman(x, y) -> float:
    """Spearman's rho: Pearson's r of the ranks, ties sharing their mean rank."""
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
