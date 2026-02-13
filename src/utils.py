# src/utils.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np


@dataclass
class DetectedSchema:
    entity_col: str
    exposure_col: Optional[str]
    primary_score_col: Optional[str]
    score_cols: List[str]
    event_cols: List[str]
    dim_cols: List[str]
    time_range: Optional[Tuple[str, str]]


def _normalize_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip()).lower()


def detect_header_row(df_raw: pd.DataFrame, required_tokens=("driver", "safety")) -> int:
    """
    For Excel sheets with title rows above the real header.
    Finds a row that likely contains column headers.
    """
    best_i = 0
    best_score = -1

    for i in range(min(len(df_raw), 50)):
        row = df_raw.iloc[i].astype(str).fillna("")
        joined = " | ".join(row.tolist()).lower()

        score = sum(tok in joined for tok in required_tokens)
        # Bonus for having many non-empty distinct "cells"
        non_empty = (row.str.strip() != "").sum()
        score += int(non_empty >= 6)

        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def load_report(file) -> pd.DataFrame:
    """
    Supports CSV/XLSX. Returns a cleaned dataframe with proper headers.
    """
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
        return _clean_df(df)

    # Excel
    df_raw = pd.read_excel(file, sheet_name=0, header=None, engine="openpyxl")
    hdr = detect_header_row(df_raw)
    df = pd.read_excel(file, sheet_name=0, header=hdr, engine="openpyxl")
    return _clean_df(df)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop empty columns
    df = df.dropna(axis=1, how="all")
    # Drop empty rows
    df = df.dropna(axis=0, how="all")

    # Strip strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    return df


def detect_schema(df: pd.DataFrame) -> DetectedSchema:
    cols = list(df.columns)
    ncols = [_normalize_col(c) for c in cols]

    # Entity
    entity_candidates = [c for c, nc in zip(cols, ncols) if "driver" in nc]
    entity_col = entity_candidates[0] if entity_candidates else cols[0]

    # Exposure
    exposure_candidates = [c for c, nc in zip(cols, ncols) if "distance" in nc or "miles" in nc or "km" in nc]
    exposure_col = exposure_candidates[0] if exposure_candidates else None

    # Scores
    score_cols = [c for c, nc in zip(cols, ncols) if "score" in nc]
    primary_score_candidates = [c for c in score_cols if "safety" in _normalize_col(c)]
    primary_score_col = primary_score_candidates[0] if primary_score_candidates else (score_cols[0] if score_cols else None)

    # Events (columns like "HB #", "Idle #", "PSL #", etc.)
    event_cols = [c for c, nc in zip(cols, ncols) if "#" in str(c) or nc.endswith(" #") or " events" in nc]
    # Remove obvious non-event numeric columns
    if exposure_col in event_cols:
        event_cols.remove(exposure_col)

    # Dimensions: non-numeric excluding entity
    dim_cols = []
    for c in cols:
        if c == entity_col:
            continue
        if df[c].dtype == object:
            dim_cols.append(c)

    return DetectedSchema(
        entity_col=entity_col,
        exposure_col=exposure_col,
        primary_score_col=primary_score_col,
        score_cols=score_cols,
        event_cols=event_cols,
        dim_cols=dim_cols,
        time_range=None,
    )


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def safe_percentile(s: pd.Series, p: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return float("nan")
    return float(np.percentile(s, p))


def format_missingness(df: pd.DataFrame) -> list:
    res = []
    for c in df.columns:
        miss = float(df[c].isna().mean() * 100.0)
        if miss >= 5:
            res.append({"column": c, "missing_pct": round(miss, 1)})
    return sorted(res, key=lambda x: -x["missing_pct"])
