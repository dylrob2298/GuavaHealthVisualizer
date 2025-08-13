import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Optional
from dateutil.parser import parse

# Prefer lxml for speed; fall back to stdlib
try:
    from lxml import etree as ET
    _USE_LXML = True
except Exception:
    import xml.etree.ElementTree as ET
    _USE_LXML = False

MOBILITY_AID_TYPES = {"Walker", "Manual Wheelchair", "Automatic Wheelchair", "Powered Wheelchair"}
LOCAL_TZ = "Europe/Madrid"
_LOCAL_TZ = ZoneInfo(LOCAL_TZ)

SLEEP_IDENTIFIER = "HKCategoryTypeIdentifierSleepAnalysis"
HR_IDENTIFIER = "HKQuantityTypeIdentifierHeartRate"
WALKING_HR_IDENTIFIER = "HKQuantityTypeIdentifierWalkingHeartRateAverage"
STEPS_IDENTIFIER = "HKQuantityTypeIdentifierStepCount"
EXERCISE_IDENTIFIER = "HKQuantityTypeIdentifierAppleExerciseTime"
TYPES_OF_INTEREST = {
    SLEEP_IDENTIFIER, HR_IDENTIFIER, WALKING_HR_IDENTIFIER, STEPS_IDENTIFIER, EXERCISE_IDENTIFIER
}

ROUND2_COLS = ["Exercise Minutes", "Sleep Hours", "Walking HR Avg", "Heart Rate Max"]



def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    return df


def prepare_symptoms_column(df: pd.DataFrame) -> pd.Series:
    symptoms = (
        df[df["type"] == "Symptoms"]
        .groupby("datetime")["name"]
        .apply(lambda s: sorted({x for x in s.dropna()}))
        .rename("Symptoms")
    )
    return symptoms


def prepare_energy_column(df: pd.DataFrame) -> pd.Series:
    energy = (df[df["type"] == "Energy"]
              .dropna(subset=["datetime", "value"])
              .groupby("datetime")["value"].last()
              .rename("Energy")
            )
    energy = pd.to_numeric(energy, errors="coerce")
    return energy


def prepare_water_column(df: pd.DataFrame) -> pd.Series:
    water_df = df[df["type"] == "Water"].copy()

    # Normalize dtypes to avoid incompatible partial assignment
    water_df["value"] = pd.to_numeric(water_df["value"], errors="coerce").astype("float64")
    water_df["unit"] = water_df["unit"].astype("string")

    # Convert any non-"mL" unit (assumed [foz_us]) to mL
    mask_not_ml = water_df["unit"].ne("mL") & water_df["unit"].notna()
    conv = 29.5735
    water_df["value_ml"] = np.where(mask_not_ml, water_df["value"] * conv, water_df["value"]).round(2)
    water_df.loc[mask_not_ml, "unit"] = "mL"

    # Pivot to a single "Water" column in mL
    water = water_df.pivot(index="datetime", columns="type", values="value_ml")
    water = water.rename(columns={"Water": "Water"})  # keeps column name explicit if desired
    return water



def prepare_mobility_aid_column(df: pd.DataFrame) -> pd.Series:
    mobility_aid = (
        df[df["type"].isin(MOBILITY_AID_TYPES)]
        .groupby("datetime")["type"]
        .apply(lambda s: sorted({x for x in s.dropna()}))
        .rename("Mobility Aids")
    )
    return mobility_aid

def _parse_apple_ts_local(ts: str) -> datetime | None:
    """
    Parses various timestamp formats and converts to the local timezone.
    """
    if not ts:
        return None
    try:
        dt = parse(ts) # aware in source offset
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_LOCAL_TZ)
    except (ValueError, TypeError):
        return None

def _accumulate_minutes_across_days(start: datetime, end: datetime, add_minutes):
    cur = start
    while cur.date() < end.date():
        next_midnight = datetime.combine(cur.date() + timedelta(days=1), datetime.min.time(), tzinfo=cur.tzinfo)
        minutes = (next_midnight - cur).total_seconds() / 60.0
        if minutes > 0:
            add_minutes(cur.date(), minutes)
        cur = next_midnight
    minutes = (end - cur).total_seconds() / 60.0
    if minutes > 0:
        add_minutes(cur.date(), minutes)

def load_apple_health_daily(xml_path: str) -> pd.DataFrame:
    """
    Fast per-day aggregates from Apple Health export.xml:
      - Steps (sum)
      - Exercise Minutes (sum)
      - Sleep Hours (sum of asleep minutes / 60)
      - Walking HR Avg (mean)
      - Heart Rate Max (max)
    """
    steps_sum = defaultdict(float)
    exercise_min_sum = defaultdict(float)
    sleep_min_sum = defaultdict(float)
    walking_hr_sum = defaultdict(float)
    walking_hr_cnt = defaultdict(int)
    hr_max = defaultdict(lambda: -np.inf)

    # iterparse setup
    if _USE_LXML:
        context = ET.iterparse(xml_path, events=("end",), tag="Record")
    else:
        context = ET.iterparse(xml_path, events=("end",))

    for _, elem in context:
        if not _USE_LXML and elem.tag != "Record":
            elem.clear()
            continue

        rtype = elem.attrib.get("type")
        if rtype not in TYPES_OF_INTEREST:
            # cheap clear
            if _USE_LXML:
                elem.clear()
                parent = elem.getparent()
                if parent is not None:
                    while elem.getprevious() is not None:
                        del parent[0]
            else:
                elem.clear()
            continue

        value = elem.attrib.get("value")
        start = elem.attrib.get("startDate")
        end = elem.attrib.get("endDate")

        # Quantity samples -> assign to local day of end (or start)
        if rtype in {STEPS_IDENTIFIER, EXERCISE_IDENTIFIER, HR_IDENTIFIER, WALKING_HR_IDENTIFIER}:
            ts = _parse_apple_ts_local(end or start)
            if ts is not None:
                day = ts.date()
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    v = np.nan
                if np.isfinite(v):
                    if rtype == STEPS_IDENTIFIER:
                        steps_sum[day] += v
                    elif rtype == EXERCISE_IDENTIFIER:
                        exercise_min_sum[day] += v  # value is minutes
                    elif rtype == HR_IDENTIFIER:
                        if v > hr_max[day]:
                            hr_max[day] = v
                    elif rtype == WALKING_HR_IDENTIFIER:
                        walking_hr_sum[day] += v
                        walking_hr_cnt[day] += 1

        # Sleep intervals -> split across midnights
        elif rtype == SLEEP_IDENTIFIER and start and end:
            s = _parse_apple_ts_local(start)
            e = _parse_apple_ts_local(end)
            if s and e and e > s:
                val = (value or "").lower()
                is_asleep = "asleep" in val  # counts Core/Deep/REM/Unspecified; ignores InBed
                if is_asleep:
                    def add_minutes(d, mins): sleep_min_sum[d] += mins
                    _accumulate_minutes_across_days(s, e, add_minutes)

        # deep clear to keep memory low
        if _USE_LXML:
            elem.clear()
            # Also eliminate now-empty references from the root node to elem
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]
        else:
            elem.clear()

    # Build DataFrame
    all_days = set(steps_sum) | set(exercise_min_sum) | set(sleep_min_sum) | set(walking_hr_cnt) | set(hr_max)
    rows = []
    for day in sorted(all_days):
        rec = {"Date": pd.to_datetime(day).date()}
        if day in steps_sum: rec["Steps"] = int(round(steps_sum[day]))
        if day in exercise_min_sum: rec["Exercise Minutes"] = exercise_min_sum[day]
        if day in sleep_min_sum: rec["Sleep Hours"] = sleep_min_sum[day] / 60.0
        if walking_hr_cnt.get(day, 0) > 0:
            rec["Walking HR Avg"] = walking_hr_sum[day] / walking_hr_cnt[day]
        if hr_max.get(day, -np.inf) != -np.inf:
            rec["Heart Rate Max"] = hr_max[day]
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["Date"])
    daily = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    # round to 2 decimals; keep steps as integers
    for c in ["Exercise Minutes", "Sleep Hours", "Walking HR Avg", "Heart Rate Max"]:
        if c in daily.columns:
            daily[c] = pd.to_numeric(daily[c], errors="coerce").round(2)
    if "Steps" in daily.columns:
        daily["Steps"] = pd.to_numeric(daily["Steps"], errors="coerce").round(0).astype("Int64")

    return daily


def transform_data(df: pd.DataFrame, apple_daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    base_datetime_index = pd.Index(df["datetime"].dropna().unique(), name="datetime")
    base_datetime_index = pd.Index(sorted(base_datetime_index))
    out_df = pd.DataFrame(index=base_datetime_index)

    symptoms = prepare_symptoms_column(df)
    energy = prepare_energy_column(df)
    water = prepare_water_column(df)
    mobility_aid = prepare_mobility_aid_column(df)

    out_df = out_df.join(symptoms, how="left")
    # Add total symptoms count column
    out_df["Total Symptoms"] = out_df["Symptoms"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    out_df["Any Symptoms"] = out_df["Total Symptoms"] > 0


    out_df = out_df.join(energy, how="left")
    out_df = out_df.join(water, how="left")
    out_df = out_df.join(mobility_aid, how="left")

    out_df["Any Mobility Aid Used"] = out_df["Mobility Aids"].apply(lambda x: True if isinstance(x, list) and len(x) > 0 else False)

    dt = pd.to_datetime(out_df.index, errors="coerce")
    out_df.insert(0, "Datetime", dt)  # keep for plotting on a time axis
    out_df.insert(1, "Date", dt.date)
    out_df["Day of Week"] = dt.day_name()
    out_df["Month"] = dt.month_name()

    # Merge Apple Health by Date
    if apple_daily is not None and not apple_daily.empty:
        apple_daily = apple_daily.copy()
        apple_daily["Date"] = pd.to_datetime(apple_daily["Date"]).dt.date
        out_df = out_df.join(apple_daily.set_index("Date"), on="Date", how="left")

    # enforce 2-dec rounding on merged Apple metrics
    for c in ROUND2_COLS:
        if c in out_df.columns:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").round(2)

    out_df = out_df.dropna(subset=["Datetime", "Date"])

    return out_df

METRICS = ["Steps","Exercise Minutes","Sleep Hours","Walking HR Avg","Heart Rate Max","Water"]

# ---------- Prep ----------
def prep_daily(d: pd.DataFrame) -> pd.DataFrame:
    df = d.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    assert df["Datetime"].is_unique, "Expected one row per day."
    return df

def _present_metrics(df: pd.DataFrame, base=METRICS):
    return [c for c in base if c in df.columns]

# ---------- Static correlations ----------
def corr_tables(df: pd.DataFrame, targets=("Energy","Total Symptoms"), method=("pearson","spearman")):
    present = _present_metrics(df)
    out = {}
    for t in targets:
        if t not in df.columns: continue
        sub = df[[t] + present].copy()
        for m in method:
            out[(t, m)] = sub.corr(method=m).loc[present, t].rename(f"{t} ({m})").to_frame()
    if not out: return pd.DataFrame()
    res = pd.concat(out.values(), axis=1)
    return res.round(3)

def plot_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    table = corr_tables(df)
    if table.empty: return go.Figure()
    fig = px.imshow(
        table.T, text_auto=True, aspect="auto",
        title="Correlations with Energy and Total Symptoms"
    )
    fig.update_layout(margin=dict(l=40,r=20,t=50,b=50))
    return fig

# ---------- Scatter grids (relationship shape) ----------
def _scatter_grid(df: pd.DataFrame, target: str) -> go.Figure:
    present = _present_metrics(df)
    frames = []
    for m in present:
        if m not in df.columns:
            continue
        tmp = df[[target, m, "Datetime"]].dropna().rename(columns={m: "value"})
        tmp["metric"] = m
        frames.append(tmp)
    if not frames:
        return go.Figure()

    long = pd.concat(frames, ignore_index=True)

    fig = px.scatter(
        long,
        x=target, y="value",
        facet_col="metric", facet_col_wrap=3,
        hover_data={"Datetime": True, target: ':.2f', "value": ':.2f'},
        title=f"{target} vs other metrics",
        render_mode="webgl",
    )
    # Independent y-axes per facet and cleaner titles
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    fig.update_traces(marker=dict(size=6), opacity=0.7)
    fig.update_layout(margin=dict(l=40, r=20, t=50, b=60))
    return fig


def plot_energy_scatter_grid(df: pd.DataFrame) -> go.Figure:
    if "Energy" not in df: return go.Figure()
    return _scatter_grid(df, "Energy")

def plot_symptoms_scatter_grid(df: pd.DataFrame) -> go.Figure:
    if "Total Symptoms" not in df: return go.Figure()
    return _scatter_grid(df, "Total Symptoms")

# ---------- Rolling correlations over time ----------
def plot_rolling_corr(df: pd.DataFrame, target="Energy", window=21, method="spearman") -> go.Figure:
    if target not in df: return go.Figure()
    present = _present_metrics(df)
    lines = []
    for m in present:
        sub = df[[target, m]].copy().dropna()
        if len(sub) < 3: continue
        rc = sub[target].rolling(window=window).corr(sub[m])
        lines.append(pd.DataFrame({"Datetime": sub.index if isinstance(sub.index, pd.DatetimeIndex) else None,
                                   "idx": np.arange(len(sub)), "metric": m, "corr": rc.values}))
    if not lines: return go.Figure()
    # Use Datetime from original df for x
    out = []
    for m in present:
        sub = df[["Datetime", target, m]].dropna().copy().reset_index(drop=True)
        if len(sub) < 3: continue
        sub["corr"] = sub[target].rolling(window=window).corr(sub[m])
        sub["metric"] = m
        out.append(sub[["Datetime","metric","corr"]])
    long = pd.concat(out, ignore_index=True).dropna(subset=["corr"])
    fig = px.line(long, x="Datetime", y="corr", color="metric",
                  title=f"Rolling {method} correlation ({window}d) with {target}")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=50,b=50))
    return fig

# ---------- Lag correlations (lead/lag) ----------
def lag_corr(df: pd.DataFrame, target: str, metric: str, max_lag=14, method="spearman"):
    x = df[target]
    y = df[metric]
    lags, vals = [], []
    for k in range(-max_lag, max_lag+1):
        if k < 0:
            corr = x.shift(-k).corr(y, method=method)
        else:
            corr = x.corr(y.shift(k), method=method)
        lags.append(k)
        vals.append(corr)
    return pd.DataFrame({"lag": lags, "corr": vals})

def plot_lag_corr_all(df: pd.DataFrame, target="Energy", max_lag=14, method="spearman") -> go.Figure:
    present = _present_metrics(df)
    frames = []
    for m in present:
        sub = lag_corr(df[[target, m]].dropna(), target, m, max_lag=max_lag, method=method)
        sub["metric"] = m
        frames.append(sub)
    if not frames: return go.Figure()
    long = pd.concat(frames, ignore_index=True)
    fig = px.line(long, x="lag", y="corr", color="metric",
                  title=f"Lag correlation with {target} (negative=metric leads)")
    fig.add_vline(x=0, line_dash="dot")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(margin=dict(l=40,r=20,t=50,b=50))
    return fig

# ---------- Normalized timeline with symptom flag ----------
def plot_normalized_timeline(df: pd.DataFrame, cols=None) -> go.Figure:
    if cols is None:
        cols = _present_metrics(df) + [c for c in ["Energy","Total Symptoms"] if c in df]
    cols = [c for c in cols if c in df.columns]
    if not cols: return go.Figure()
    z = df[cols].apply(lambda s: (s - s.mean())/s.std(ddof=0), axis=0)
    z["Datetime"] = df["Datetime"]
    long = z.melt(id_vars="Datetime", var_name="metric", value_name="z")
    fig = px.line(long, x="Datetime", y="z", color="metric",
                  title="Standardized metrics over time (z-scores)")
    # Add symptom flag band if available
    if "Any Symptoms" in df.columns:
        sym = df[["Datetime","Any Symptoms"]].copy()
        sym["flag"] = sym["Any Symptoms"].astype(int)
        fig.add_trace(
            go.Bar(x=sym["Datetime"], y=sym["flag"]*0.2, name="Symptoms present",
                   marker_opacity=0.25, yaxis="y2", showlegend=True)
        )
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0,1], showticklabels=False))
    fig.update_layout(hovermode="x unified", margin=dict(l=40,r=20,t=50,b=50))
    return fig

# ---------- Symptom presence heatmap over time ----------
def plot_symptom_heatmap(df: pd.DataFrame, top_n=20) -> go.Figure:
    if "Symptoms" not in df: return go.Figure()
    # Count frequency and keep top N
    from collections import Counter
    cnt = Counter()
    for lst in df["Symptoms"].dropna():
        if isinstance(lst, list): cnt.update(set(lst))
    if not cnt: return go.Figure()
    top = {k for k,_ in cnt.most_common(top_n)}
    # Build presence matrix
    rows = []
    for i, row in df[["Datetime","Symptoms"]].iterrows():
        date = row["Datetime"]
        lst = row["Symptoms"] if isinstance(row["Symptoms"], list) else []
        for s in top:
            rows.append({"Datetime": date, "Symptom": s, "Present": int(s in lst)})
    mat = pd.DataFrame(rows)
    if mat.empty: return go.Figure()
    pivot = mat.pivot(index="Symptom", columns="Datetime", values="Present").fillna(0)
    fig = px.imshow(pivot, aspect="auto", title="Symptom presence over time (top)")
    fig.update_layout(margin=dict(l=40,r=20,t=50,b=50))
    return fig

def main():
    print("Hello from guavahealthvisualizer!")
    df_raw = load_csv("guava.csv")

    # Load Apple Health aggregates from export.xml
    try:
        apple_daily = load_apple_health_daily("export.xml")
    except FileNotFoundError:
        apple_daily = pd.DataFrame(columns=["Date"])

    transformed = transform_data(df_raw, apple_daily=apple_daily)
    print(transformed)

    transformed.info()
    transformed.to_csv("transformed_guava.csv", index=False)

    df = prep_daily(transformed)

    figs = {
        "corr_heatmap": plot_corr_heatmap(df),
        "energy_vs_all": plot_energy_scatter_grid(df),
        "symptoms_vs_all": plot_symptoms_scatter_grid(df),
        "rolling_corr_energy": plot_rolling_corr(df, target="Energy", window=21),
        "lag_corr_energy": plot_lag_corr_all(df, target="Energy", max_lag=14),
        "normalized_timeline": plot_normalized_timeline(df),
        "symptom_heatmap": plot_symptom_heatmap(df, top_n=15),
    }
    for name, fig in figs.items():
        fig.show()


if __name__ == "__main__":
    main()
