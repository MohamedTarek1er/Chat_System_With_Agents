import re
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import tool

# numeric-looking columns that are identifiers/axes, not metrics to analyse
_NON_METRIC_NAMES = {"year", "epoch", "epochs", "page", "rank", "id", "index"}


def _clean_value(val):
    """Turn '92%', '$1,200', '0.85' into floats where possible."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        s = re.sub(r"[,%$]", "", val.strip())
        try:
            return float(s)
        except ValueError:
            return val
    return val


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    cleaned = series.map(_clean_value)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    # numeric column only if at least half of the non-null values parsed
    if numeric.notna().sum() >= 0.5 * cleaned.notna().sum():
        return numeric
    return cleaned


def _build_dataframe(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.DataFrame(records)
    for col in df.columns:
        df[col] = _to_numeric_clean(df[col])
    return df, df.select_dtypes(include="number").columns.tolist()


# ---------------------------------------------------------------- analyses
def _summary(df, cols, labels) -> str:
    lines = ["### Distribution & Summary Statistics"]
    for col in cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, med, q3 = s.quantile([0.25, 0.50, 0.75])
        std = f"{s.std():.4f}" if len(s) > 1 else "N/A"
        lines.append(
            f"- **{col}** (n={len(s)}): mean={s.mean():.4f}, median={med:.4f}, std={std}, "
            f"min={s.min():.4f} ({labels[s.idxmin()]}), max={s.max():.4f} ({labels[s.idxmax()]}), "
            f"Q1={q1:.4f}, Q3={q3:.4f}, range={s.max() - s.min():.4f}"
        )
    return "\n".join(lines)


def _rank(df, cols, labels, lower_is_better) -> str:
    work = df.copy()
    work["_label"] = labels

    if len(cols) == 1:
        col = cols[0]
        ascending = col in lower_is_better
        ranked = work.dropna(subset=[col]).sort_values(by=col, ascending=ascending)
        if ranked.empty:
            return f"Error: no values in '{col}'."
        best = ranked[col].iloc[0]
        lines = [f"### Ranking by {col} ({'lower' if ascending else 'higher'} is better)"]
        for i, (_, row) in enumerate(ranked.iterrows(), start=1):
            diff = row[col] - best
            pct = f" ({diff / best * 100:+.2f}%)" if best else ""
            lines.append(f"{i}. **{row['_label']}**: {row[col]:.4f} | diff vs #1: {diff:+.4f}{pct}")
        return "\n".join(lines)

    # multi-metric: min-max normalise each metric to [0, 1], then average
    complete = work.dropna(subset=cols).copy()
    if complete.empty:
        return "Error: no row has values for all requested metrics."
    dropped = work.loc[~work.index.isin(complete.index), "_label"].tolist()

    norm_cols = []
    for col in cols:
        lo, hi = complete[col].min(), complete[col].max()
        if hi == lo:
            scaled = pd.Series(1.0, index=complete.index)
        else:
            scaled = (complete[col] - lo) / (hi - lo)
            if col in lower_is_better:
                scaled = 1.0 - scaled
        complete[f"_n_{col}"] = scaled
        norm_cols.append(f"_n_{col}")

    complete["_overall"] = complete[norm_cols].mean(axis=1)
    ranked = complete.sort_values(by="_overall", ascending=False)

    lines = [f"### Overall ranking across {', '.join(cols)} (normalised 0-1 score)"]
    for i, (_, row) in enumerate(ranked.iterrows(), start=1):
        metrics = ", ".join(f"{c}={row[c]:.4f}" for c in cols)
        lines.append(f"{i}. **{row['_label']}** - overall={row['_overall']:.4f} ({metrics})")
    if dropped:
        lines.append(f"_Excluded (missing metric values): {', '.join(map(str, dropped))}_")
    return "\n".join(lines)


def _differences(df, cols, labels) -> str:
    lines = ["### Differences & Percentage Gaps"]
    for col in cols:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        mean = s.mean()
        spread = s.max() - s.min()
        spread_pct = f" ({spread / s.min() * 100:.2f}%)" if s.min() else ""
        lines.append(
            f"- **{col}**: max-min gap = {spread:.4f}{spread_pct} "
            f"[{labels[s.idxmax()]}={s.max():.4f} vs {labels[s.idxmin()]}={s.min():.4f}]"
        )
        for idx, val in s.items():
            vs_mean = val - mean
            pct = f" ({vs_mean / mean * 100:+.2f}%)" if mean else ""
            lines.append(f"  - {labels[idx]}: {val:.4f} | vs mean ({mean:.4f}): {vs_mean:+.4f}{pct}")
    return "\n".join(lines)


def _relationships(df, cols, labels) -> str:
    if len(df) < 2:
        return "Need at least 2 rows to analyse trends or relationships."
    lines = ["### Trends & Variable Relationships"]

    for col in cols:  # sequential trend (row order = years/epochs/versions)
        s = df[col].dropna()
        if len(s) < 2:
            continue
        delta = s.iloc[-1] - s.iloc[0]
        pct = f" ({delta / s.iloc[0] * 100:+.2f}%)" if s.iloc[0] else ""
        direction = "increasing" if delta > 0 else "decreasing" if delta < 0 else "flat"
        lines.append(
            f"- **{col} trend** ({labels[s.index[0]]} -> {labels[s.index[-1]]}): "
            f"{direction}, net change = {delta:+.4f}{pct}"
        )

    if len(cols) >= 2 and len(df) >= 3:
        corr = df[cols].corr()
        lines.append(f"- **Pairwise correlations (Pearson r, n={len(df)})**:")
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                r = corr.loc[c1, c2]
                if pd.notna(r):
                    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
                    lines.append(f"  - {c1} vs {c2}: r={r:.4f} ({strength} {'positive' if r > 0 else 'negative'})")
        if len(df) < 5:
            lines.append("  _Caution: fewer than 5 rows, correlations are not reliable._")
    return "\n".join(lines)


# -------------------------------------------------------------------- tool
@tool
def data_analysis(
    records: list[dict[str, Any]],
    analysis_type: Literal["summary", "rank", "differences", "relationships"] = "summary",
    label_column: str | None = None,
    metric_columns: list[str] | None = None,
    lower_is_better: list[str] | None = None,
) -> str:
    """Quantitative analysis of numbers taken from the retrieved evidence or extracted tables.

    Rules for `records`: copy values exactly as they appear in the evidence, use the same
    scale within a column (all percentages as 92, or all as 0.92), and include a label
    column (e.g. "model" or "paper") identifying each row.

    Args:
        records: Row dicts, e.g. [{"model": "CNN-A", "accuracy": 92, "loss": 0.31}, ...].
        analysis_type:
            - "summary": count, mean, median, std, min, max, quartiles per metric.
            - "rank": rank rows by one metric, or by a normalised overall score across several.
            - "differences": max-min gap and each row's difference vs the mean.
            - "relationships": first-to-last trend and pairwise correlations.
        label_column: Column identifying each row (auto-detected if omitted).
        metric_columns: Numeric columns to analyse. Pass this whenever the records contain
            year/epoch/page-like columns. Defaults to all numeric columns.
        lower_is_better: Metrics where a lower value is better, e.g. ["loss", "latency", "error"].
    """
    if not records:
        return "Error: 'records' cannot be empty."

    df, all_numeric = _build_dataframe(records)

    if metric_columns:
        cols = [c for c in metric_columns if c in all_numeric]
    else:
        cols = [c for c in all_numeric if c.lower() not in _NON_METRIC_NAMES]
    if not cols:
        return "Error: no numeric columns found in records."

    if not label_column or label_column not in df.columns:
        text_cols = [c for c in df.columns if c not in all_numeric]
        label_column = text_cols[0] if text_cols else None
    labels = (df[label_column].astype(str).tolist() if label_column 
            else [f"Row {i + 1}" for i in range(len(df))])

    lower = set(lower_is_better or [])

    if analysis_type == "summary":
        return _summary(df, cols, labels)
    if analysis_type == "rank":
        return _rank(df, cols, labels, lower)
    if analysis_type == "differences":
        return _differences(df, cols, labels)
    if analysis_type == "relationships":
        return _relationships(df, cols, labels)
    return f"Error: unsupported analysis_type '{analysis_type}'."
