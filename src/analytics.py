from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .utils import DetectedSchema, coerce_numeric, safe_percentile, format_missingness


def compute_facts_pack(df: pd.DataFrame, schema: DetectedSchema) -> Dict[str, Any]:
    df = df.copy()

    numeric_cols: List[str] = []
    for c in (schema.score_cols + schema.event_cols + ([schema.exposure_col] if schema.exposure_col else [])):
        if c and c in df.columns and c not in numeric_cols:
            numeric_cols.append(c)

    df = coerce_numeric(df, numeric_cols)

    event_cols = [c for c in schema.event_cols if c in df.columns]
    if event_cols:
        df["total_events"] = df[event_cols].fillna(0).sum(axis=1)
    else:
        df["total_events"] = np.nan

    # Normalized rate per 100 exposure
    if schema.exposure_col and schema.exposure_col in df.columns:
        exposure = df[schema.exposure_col].replace({0: np.nan})
        df["events_per_100_exposure"] = (df["total_events"] / exposure) * 100.0
    else:
        df["events_per_100_exposure"] = np.nan

    entity = schema.entity_col
    primary = schema.primary_score_col
    exposure_col = schema.exposure_col

    def top_list(col: str, asc: bool, n=5):
        if col is None or col not in df.columns:
            return []
        x = df[[entity, col]].dropna().sort_values(col, ascending=asc).head(n)
        return [{"entity": str(r[entity]), "value": float(r[col])} for _, r in x.iterrows()]

    leaderboards = {
        "lowest_primary_score": top_list(primary, asc=True, n=5) if primary else [],
        "highest_total_events": top_list("total_events", asc=False, n=5),
        "highest_normalized_event_rate": top_list("events_per_100_exposure", asc=False, n=5),
    }

    distribution = {}
    if primary and primary in df.columns:
        distribution["primary_score"] = {
            "min": float(np.nanmin(df[primary])),
            "p10": safe_percentile(df[primary], 10),
            "median": safe_percentile(df[primary], 50),
            "p90": safe_percentile(df[primary], 90),
            "max": float(np.nanmax(df[primary])),
        }
    distribution["total_events"] = {
        "min": float(np.nanmin(df["total_events"])),
        "p90": safe_percentile(df["total_events"], 90),
        "max": float(np.nanmax(df["total_events"])),
    }

    # Pareto summary stats
    pareto_items = []
    for metric in ["total_events"] + event_cols:
        s = df[[entity, metric]].dropna().sort_values(metric, ascending=False)
        if len(s) < 3:
            continue
        total = s[metric].sum()
        if total <= 0:
            continue
        top_20_n = max(1, int(np.ceil(0.2 * len(s))))
        top_20_share = float(s.head(top_20_n)[metric].sum() / total)
        top_10_share = float(s.head(min(10, len(s)))[metric].sum() / total)
        pareto_items.append(
            {
                "metric": metric,
                "top_20pct_contribution": round(top_20_share, 3),
                "top_10_entities_contribution": round(top_10_share, 3),
            }
        )

    # Relationships
    relationships = []
    if exposure_col and exposure_col in df.columns:
        if df[exposure_col].notna().sum() >= 5 and df["total_events"].notna().sum() >= 5:
            corr = float(pd.concat([df[exposure_col], df["total_events"]], axis=1).corr().iloc[0, 1])
            if np.isfinite(corr):
                relationships.append(
                    {
                        "x": exposure_col,
                        "y": "total_events",
                        "method": "pearson",
                        "value": round(corr, 3),
                        "note": "Correlation is indicative; interpret with context.",
                    }
                )

    # Data quality
    data_quality = {
        "missingness": format_missingness(df),
        "type_issues": [],
        "suspicious_entities": [],
    }

    ent_vals = df[entity].astype(str).fillna("")
    for v in ent_vals.head(50):
        vv = v.strip()
        if vv == "":
            continue
        if vv.isdigit():
            data_quality["suspicious_entities"].append({"value": vv, "reason": "Entity appears numeric-only."})
        if len(vv) <= 2:
            data_quality["suspicious_entities"].append({"value": vv, "reason": "Entity value is unusually short."})

    # ✅ NEW: chart-ready datasets
    top_risky_entities = _top_risky_entities(df, schema, n=50)   # include more rows for charts
    exposure_vs_events = _exposure_vs_events(df, schema)         # for scatter chart
    pareto_curve_total_events = _pareto_curve(df, schema, metric="total_events")

    chart_data = {
        "top_risky_entities": top_risky_entities,
        "exposure_vs_events": exposure_vs_events,
        "pareto_curve_total_events": pareto_curve_total_events,
        "df_preview": df.head(200).to_dict(orient="records"),
        "columns": list(df.columns),
    }

    return {
        "dataset_profile": {
            "report_name": "Uploaded Report",
            "entity_type": "driver",
            "time_range": {"from": None, "to": None},
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "dimensions": [schema.entity_col] + schema.dim_cols,
            "metrics": {
                "scores": schema.score_cols,
                "event_counts": schema.event_cols,
                "exposure": [schema.exposure_col] if schema.exposure_col else [],
            },
            "mappings": {
                "entity_name": schema.entity_col,
                "exposure": schema.exposure_col,
                "primary_score": schema.primary_score_col,
            },
        },
        "data_quality": data_quality,
        "computed_fields": {
            "total_events_field": "total_events",
            "normalized_event_rate_field": "events_per_100_exposure",
            "normalization_exposure_unit": "exposure",
        },
        "leaderboards": leaderboards,
        "distribution": distribution,
        "pareto": pareto_items,
        "relationships": relationships,
        "chart_data": chart_data,
    }


def _top_risky_entities(df: pd.DataFrame, schema: DetectedSchema, n=50) -> list:
    entity = schema.entity_col
    primary = schema.primary_score_col
    exposure_col = schema.exposure_col

    cols = [entity, "total_events", "events_per_100_exposure"]
    if primary and primary in df.columns:
        cols.append(primary)
    if exposure_col and exposure_col in df.columns:
        cols.append(exposure_col)

    x = df[cols].copy().dropna(subset=[entity])

    # Risk rank: normalized rate first (if available), then low safety score, then total_events
    if x["events_per_100_exposure"].notna().any():
        x["_risk"] = x["events_per_100_exposure"].fillna(-1)
    else:
        x["_risk"] = x["total_events"].fillna(-1)

    if primary and primary in x.columns:
        x["_risk"] = x["_risk"] + (x[primary].max() - x[primary]).fillna(0)

    x = x.sort_values("_risk", ascending=False).head(n)

    out = []
    for _, r in x.iterrows():
        out.append(
            {
                "entity": str(r[entity]),
                "primary_score": float(r[primary]) if primary and primary in x.columns and pd.notna(r[primary]) else None,
                "total_events": float(r["total_events"]) if pd.notna(r["total_events"]) else None,
                "events_per_100_exposure": float(r["events_per_100_exposure"]) if pd.notna(r["events_per_100_exposure"]) else None,
                # ✅ NEW: include exposure for plotting
                "Distance Travelled": float(r[exposure_col]) if exposure_col and exposure_col in x.columns and pd.notna(r[exposure_col]) else None,
            }
        )
    return out


def _exposure_vs_events(df: pd.DataFrame, schema: DetectedSchema) -> list:
    """
    Chart dataset for scatter: exposure vs total_events (all drivers).
    Produces columns that match typical chart specs: 'Distance Travelled' and 'total_events'.
    """
    exposure_col = schema.exposure_col
    entity = schema.entity_col

    if not exposure_col or exposure_col not in df.columns:
        return []

    x = df[[entity, exposure_col, "total_events"]].copy()
    x = x.dropna(subset=[entity, exposure_col, "total_events"])
    x = x.rename(columns={exposure_col: "Distance Travelled"})
    # keep all rows, but cap to avoid huge payload to LLM/UI
    x = x.head(500)

    return x.to_dict(orient="records")


def _pareto_curve(df: pd.DataFrame, schema: DetectedSchema, metric: str = "total_events") -> list:
    """
    Builds a pareto curve dataset:
    - rank: 1..N
    - cumulative_share: 0..1
    """
    entity = schema.entity_col
    if metric not in df.columns:
        return []

    s = df[[entity, metric]].dropna().sort_values(metric, ascending=False)
    if len(s) < 3:
        return []

    total = s[metric].sum()
    if total <= 0:
        return []

    s["rank"] = np.arange(1, len(s) + 1)
    s["cumulative"] = s[metric].cumsum()
    s["cumulative_share"] = s["cumulative"] / total

    # Keep a reasonable number of points
    s = s.head(200)
    return s[["rank", "cumulative_share"]].to_dict(orient="records")
