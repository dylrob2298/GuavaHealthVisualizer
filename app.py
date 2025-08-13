import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast

st.set_page_config(page_title="Health Dashboard", layout="wide")

# ---------- Load ----------
@st.cache_data
def load_data(path="transformed_guava.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").dropna(subset=["Datetime"])
    df["Date"] = df["Datetime"].dt.date

    # Parse list-like columns that were serialized to CSV
    def parse_list_cell(x):
        if isinstance(x, list): return x
        if pd.isna(x): return []
        if isinstance(x, str):
            s = x.strip()
            if not s or s.lower() == "none": return []
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list): return [str(t) for t in v]
            except Exception:
                pass
            # fallback: comma-separated
            return [t.strip() for t in s.split(",") if t.strip()]
        return []

    for col in ["Symptoms", "Mobility Aids"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_cell)

    # Normalize booleans that may have been saved as strings
    def to_bool(x):
        if isinstance(x, (bool, np.bool_)): return bool(x)
        if pd.isna(x): return False
        s = str(x).strip().lower()
        return s in {"true", "1", "yes", "y"}
    for col in ["Any Mobility Aid Used", "Any Symptoms"]:
        if col in df.columns:
            df[col] = df[col].apply(to_bool)

    if not df["Datetime"].is_unique:
        df = df.drop_duplicates(subset=["Date"], keep="last")
    return df.reset_index(drop=True)

df = load_data()

# Available columns
metrics = [c for c in ["Energy","Total Symptoms","Steps","Exercise Minutes","Sleep Hours",
                       "Walking HR Avg","Heart Rate Max","Water"] if c in df.columns]
has = {c: (c in df.columns) for c in ["Symptoms","Mobility Aids","Any Symptoms"]}

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")
date_min, date_max = df["Date"].min(), df["Date"].max()
date_range = st.sidebar.slider("Date range", value=(date_min, date_max),
                               min_value=date_min, max_value=date_max, format="YYYY-MM-DD")
window = st.sidebar.slider("Smoothing window (days)", 1, 21, 7)
show_roll = st.sidebar.checkbox("Show 7-day smoothing", True)
target_sleep = st.sidebar.number_input("Sleep target (hours)", 0.0, 24.0, 8.0, 0.5)
target_water = st.sidebar.number_input("Water target (mL)", 0.0, 10000.0, 2000.0, 100.0)
target_ex = st.sidebar.number_input("Exercise target (minutes)", 0.0, 300.0, 30.0, 5.0)

# Filter by range
mask = (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
d = df.loc[mask].copy()

# Rolling columns
for c in ["Steps","Exercise Minutes","Sleep Hours","Walking HR Avg","Energy"]:
    if c in d.columns:
        d[f"{c} (7d)"] = d[c].rolling(window, min_periods=1).mean().round(2)

# ---------- Day picker + KPI cards ----------
st.title("Your Health Data")
sel_day = st.selectbox("Pick a day to inspect", options=list(d["Date"]), index=len(d)-1)

sel = d[d["Date"] == sel_day].iloc[0] if not d.empty else None

def _fmt(x):
    if pd.isna(x): return "—"
    if isinstance(x, (int, np.integer)): return f"{int(x)}"
    try:
        f = float(x)
        return f"{f:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

cols = st.columns(6)
cards = [
    ("Energy", "Energy"),
    ("Symptoms", "Total Symptoms"),
    ("Steps", "Steps"),
    ("Sleep (hr)", "Sleep Hours"),
    ("HR Max", "Heart Rate Max"),
    ("Water (mL)", "Water"),
]
for (label, key), col in zip(cards, cols):
    val = sel[key] if sel is not None and key in d.columns else None
    col.metric(label, _fmt(val))

# Symptoms and mobility aids (chips-like text)
if sel is not None:
    left, right = st.columns(2)
    if has["Symptoms"]:
        syms = sel["Symptoms"] if isinstance(sel["Symptoms"], list) else []
        left.markdown("**Symptoms on selected day**")
        left.write(", ".join(syms) if syms else "None")
    if has["Mobility Aids"]:
        aids = sel["Mobility Aids"] if isinstance(sel["Mobility Aids"], list) else []
        right.markdown("**Mobility aids used**")
        right.write(", ".join(aids) if aids else "None")

# Reuse the sidebar smoothing window if present
win = window if "window" in globals() else 7

def add_ma(df: pd.DataFrame, col: str, window: int) -> str | None:
    if col in df.columns:
        name = f"{col} ({window}d MA)"
        df[name] = df[col].rolling(window, min_periods=1).mean()
        return name
    return None

# Use your filtered daily data
transformed = d.copy().sort_values("Datetime")
w_ma = add_ma(transformed, "Water", win)
e_ma = add_ma(transformed, "Energy", win)

# ---- Tabs (weekly patterns removed) ----
st.divider()
st.subheader("Metrics over time")

# Detect numeric metric columns (exclude booleans, dates, helpers)
exclude_cols = {"Any Symptoms", "Any Mobility Aid Used"}
base_cols = {"Datetime", "Date", "Day of Week", "Month"}
num_cols = [
    c for c in transformed.columns
    if c not in base_cols
    and "(MA)" not in c
    and c not in exclude_cols
    and pd.api.types.is_numeric_dtype(transformed[c])
    and "7d" not in c
]

# Units and targets for nicer axes and guides
unit_map = {
    "Water": "mL",
    "Exercise Minutes": "min",
    "Sleep Hours": "hr",
    "Steps": "steps",
    "Walking HR Avg": "bpm",
    "Heart Rate Max": "bpm",
    "Energy": "",
    "Total Symptoms": "count",
}
target_map = {
    "Water": float(target_water) if "target_water" in globals() else None,
    "Exercise Minutes": float(target_ex) if "target_ex" in globals() else None,
    "Sleep Hours": float(target_sleep) if "target_sleep" in globals() else None,
}

if num_cols:
    ts_tabs = st.tabs(num_cols)
    for tab, col in zip(ts_tabs, num_cols):
        with tab:
            y = transformed[col]
            fig_ts = go.Figure()
            fig_ts.add_trace(
                go.Scatter(
                    x=transformed["Datetime"], y=y, name=col,
                    mode="lines+markers", hovertemplate="<b>%{x|%Y-%m-%d}</b><br>"
                                                        f"{col}: "+"%{y:.2f}<extra></extra>"
                )
            )
            if show_roll:
                ma = y.rolling(win, min_periods=1).mean()
                fig_ts.add_trace(
                    go.Scatter(x=transformed["Datetime"], y=ma, name=f"{col} ({win}d MA)", mode="lines")
                )
            tgt = target_map.get(col)
            if tgt is not None and np.isfinite(tgt):
                fig_ts.add_hline(y=tgt, line_dash="dot",
                                 annotation_text=f"Target: {tgt:g} {unit_map.get(col,'')}".strip())

            y_title = f"{col}" + (f" ({unit_map[col]})" if unit_map.get(col) else "")
            fig_ts.update_layout(
                title=f"{col} over time",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title=y_title,
                margin=dict(l=40, r=20, t=60, b=40),
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No numeric metrics found to plot.")

# ----------------- Below: other graphs -----------------
st.divider()
st.subheader("Comparisons and summaries")

tab1, tab2, tab3, tab4 = st.tabs([
    "Water vs Energy", "Symptoms per day", "Energy + Mobility", "Top symptoms",
])

with tab1:
    if {"Water", "Energy"} <= set(transformed.columns):
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(go.Scatter(x=transformed["Datetime"], y=transformed["Water"],
                                      mode="lines+markers", name="Water (mL)"),
                           secondary_y=False)
        fig_dual.add_trace(go.Scatter(x=transformed["Datetime"], y=transformed["Energy"],
                                      mode="lines+markers", name="Energy"),
                           secondary_y=True)
        # Optional MAs
        if "Water ({}d MA)".format(win) in transformed.columns:
            fig_dual.add_trace(go.Scatter(x=transformed["Datetime"], y=transformed[f"Water ({win}d MA)"],
                                          mode="lines", name=f"Water ({win}d MA)"),
                               secondary_y=False)
        if "Energy ({}d MA)".format(win) in transformed.columns:
            fig_dual.add_trace(go.Scatter(x=transformed["Datetime"], y=transformed[f"Energy ({win}d MA)"],
                                          mode="lines", name=f"Energy ({win}d MA)"),
                               secondary_y=True)
        fig_dual.update_layout(title_text="Water vs Energy Over Time", hovermode="x unified",
                               margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", y=1.02))
        fig_dual.update_xaxes(title_text="Date")
        fig_dual.update_yaxes(title_text="Water (mL)", secondary_y=False)
        fig_dual.update_yaxes(title_text="Energy", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)
    else:
        st.info("Need both Water and Energy columns.")

with tab2:
    if "Total Symptoms" in transformed.columns:
        tmp = transformed.copy()
        color_col = "Any Mobility Aid Used" if "Any Mobility Aid Used" in tmp.columns else None
        if "Mobility Aids" in tmp.columns:
            tmp["Mobility Aids (list)"] = tmp["Mobility Aids"].apply(
                lambda v: ", ".join(map(str, v)) if v else "None"
            )
        fig_sym_ct = px.bar(
            tmp, x="Datetime", y="Total Symptoms",
            color=color_col,
            custom_data=(["Mobility Aids (list)"] if "Mobility Aids (list)" in tmp.columns else None),
            title="Total Symptoms per Day" + (" (colored by mobility aid usage)" if color_col else "")
        )
        if "Mobility Aids (list)" in tmp.columns:
            fig_sym_ct.update_traces(
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Total Symptoms: %{y}"
                              "<br>Mobility Aids: %{customdata[0]}<extra></extra>"
            )
        fig_sym_ct.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_sym_ct, use_container_width=True)
    else:
        st.info("No symptoms count available.")

with tab3:
    color_col = "Any Mobility Aid Used" if "Any Mobility Aid Used" in transformed.columns else None
    hover_cols = [c for c in ["Water","Total Symptoms","Mobility Aids"] if c in transformed.columns]
    if "Energy" in transformed.columns:
        fig_mob = px.scatter(
            transformed, x="Datetime", y="Energy",
            color=color_col, hover_data=hover_cols,
            title="Energy Over Time" + (" (colored by mobility aid usage)" if color_col else "")
        )
        fig_mob.update_traces(mode="markers")
        fig_mob.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_mob, use_container_width=True)
    else:
        st.info("Energy not available.")

with tab4:
    if "Symptoms" in transformed.columns:
        top_n = st.number_input("Top N symptoms", 5, 50, 20, 1)
        sym_long = transformed.explode("Symptoms")
        sym_counts = (
            sym_long.dropna(subset=["Symptoms"])
                    .groupby("Symptoms", as_index=False)
                    .size()
                    .sort_values("size", ascending=False)
        )
        if not sym_counts.empty:
            fig_sym_freq = px.bar(
                sym_counts.head(int(top_n)), x="size", y="Symptoms", orientation="h",
                title=f"Top {int(top_n)} Recorded Symptoms",
                labels={"size": "Days observed", "Symptoms": "Symptom"}
            )
            fig_sym_freq.update_layout(yaxis={"categoryorder": "total ascending"},
                                       margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_sym_freq, use_container_width=True)
        else:
            st.info("No symptom records in the selected range.")
    else:
        st.info("Symptoms column not available.")


