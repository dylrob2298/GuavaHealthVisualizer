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
        d[f"{c} ({window}d MA)"] = d[c].rolling(window, min_periods=1).mean().round(2)

# ---------- Day picker + KPI cards ----------
st.title("Your Health Data")
# --- MODIFICATION START ---
# Changed the selectbox to a date_input for easier navigation
sel_day = st.date_input(
    "Pick a day to inspect",
    value=d["Date"].max() if not d.empty else date_range[1],
    min_value=date_range[0],
    max_value=date_range[1],
    format="YYYY-MM-DD",
)

# Find data for the selected day. It might not exist in the filtered dataframe.
sel_df = d[d["Date"] == sel_day]
sel = sel_df.iloc[0] if not sel_df.empty else None
# --- MODIFICATION END ---

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

# Prepare a string version of Mobility Aids for clean hover display.
if has["Mobility Aids"]:
    transformed['Mobility Aids (str)'] = transformed['Mobility Aids'].apply(
        lambda aids: ', '.join(map(str, aids)) if (aids and isinstance(aids, list)) else 'None'
    )
else:
    # Create a placeholder column if the original doesn't exist for consistent hover info.
    transformed['Mobility Aids (str)'] = 'Not Tracked'

# ---- Tabs for single metrics ----
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

            hover_cols = ["Energy", "Total Symptoms", "Mobility Aids (str)"]
            available_hover_cols = [c for c in hover_cols if c in transformed.columns and c != col]
            
            customdata = transformed[available_hover_cols]

            hovertemplate = f"<b>%{{x|%Y-%m-%d}}</b><br>{col}: %{{y:.2f}}"
            for i, hover_col_name in enumerate(available_hover_cols):
                label = hover_col_name.replace(" (str)", "")
                formatter = ""
                if hover_col_name == "Energy": formatter = ":.1f"
                elif hover_col_name == "Total Symptoms": formatter = ":.0f"
                hovertemplate += f"<br>{label}: %{{customdata[{i}]{formatter}}}"
            hovertemplate += "<extra></extra>"
            
            fig_ts.add_trace(
                go.Scatter(
                    x=transformed["Datetime"], y=y, name=col,
                    mode="lines+markers",
                    customdata=customdata,
                    hovertemplate=hovertemplate
                )
            )

            if show_roll:
                ma_name = f"{col} ({win}d MA)"
                ma_col = add_ma(transformed, col, win)
                if ma_col:
                    fig_ts.add_trace(
                        go.Scatter(
                            x=transformed["Datetime"],
                            y=transformed[ma_col],
                            name=ma_name,
                            mode="lines",
                            hovertemplate=f"{ma_name}: %{{y:.2f}}<extra></extra>"
                        )
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
                legend=dict(orientation="h", y=1.02, yanchor="bottom", xanchor="right", x=1)
            )
            st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No numeric metrics found to plot.")

# --- MODIFICATION START ---
# Moved the comparison chart out of the tabs into its own section
st.divider()
st.subheader("Compare Two Metrics")

options = num_cols

if len(options) < 2:
    st.info("You need at least two numeric metric columns in your data to use this comparison chart.")
else:
    c1, c2 = st.columns(2)
    default_ix1 = options.index("Water") if "Water" in options else 0
    default_ix2 = options.index("Energy") if "Energy" in options and "Energy" != options[default_ix1] else 1

    metric1 = c1.selectbox("Select metric for Left Y-Axis", options, index=default_ix1, key="metric1_selector")
    metric2 = c2.selectbox("Select metric for Right Y-Axis", options, index=default_ix2, key="metric2_selector")
    
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    
    # --- Prepare hover data ---
    # hover_base_cols = ["Energy", "Total Symptoms", "Mobility Aids (str)"]
    # We will build the hover text from metric1, so we need metric2 and the others
    # custom_data_cols = [metric2] + [c for c in hover_base_cols if c not in [metric1, metric2]]
    # available_custom_cols = [c for c in custom_data_cols if c in transformed.columns]
    # customdata = transformed[available_custom_cols]

    # --- Build rich hovertemplate ---
    hovertemplate = f"<b>%{{x|%Y-%m-%d}}</b><br>{metric1}: %{{y:.2f}}"
    # for i, col_name in enumerate(available_custom_cols):
    #     label = col_name.replace(" (str)", "")
    #     formatter = ""
    #     if col_name == "Energy": formatter = ":.1f"
    #     elif col_name == "Total Symptoms": formatter = ":.0f"
    #     else: formatter = ":.2f" # For the other selected metric
    #     hovertemplate += f"<br>{label}: %{{customdata[{i}]{formatter}}}"
    hovertemplate += "<extra></extra>"
    
    unit1 = f" ({unit_map.get(metric1, '')})" if unit_map.get(metric1) else ""
    unit2 = f" ({unit_map.get(metric2, '')})" if unit_map.get(metric2) else ""

    # --- Add traces ---
    fig_dual.add_trace(go.Scatter(
        x=transformed["Datetime"], y=transformed[metric1],
        mode="lines+markers", name=metric1,
        customdata=customdata, hovertemplate=hovertemplate
    ), secondary_y=False)

    if show_roll:
        ma1_name = add_ma(transformed, metric1, win)
        if ma1_name:
            fig_dual.add_trace(go.Scatter(
                x=transformed["Datetime"], y=transformed[ma1_name],
                mode="lines", name=ma1_name,
                hovertemplate=f"{ma1_name}: %{{y:.2f}}<extra></extra>"
            ), secondary_y=False)
    
    fig_dual.add_trace(go.Scatter(
        x=transformed["Datetime"], y=transformed[metric2],
        mode="lines+markers", name=metric2,
        hovertemplate=f"{metric2}: %{{y:.2f}}<extra></extra>" # Minimal hover
    ), secondary_y=True)

    if show_roll:
        ma2_name = add_ma(transformed, metric2, win)
        if ma2_name:
            fig_dual.add_trace(go.Scatter(
                x=transformed["Datetime"], y=transformed[ma2_name],
                mode="lines", name=ma2_name,
                hovertemplate=f"{ma2_name}: %{{y:.2f}}<extra></extra>"
            ), secondary_y=True)

    fig_dual.update_layout(
        title_text=f"{metric1} vs. {metric2} Over Time",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", xanchor="right", x=1)
    )
    fig_dual.update_xaxes(title_text="Date")
    fig_dual.update_yaxes(title_text=f"{metric1}{unit1}", secondary_y=False)
    fig_dual.update_yaxes(title_text=f"{metric2}{unit2}", secondary_y=True)

    st.plotly_chart(fig_dual, use_container_width=True)

# ----------------- Other Summary Graphs -----------------
st.divider()
st.subheader("Other Summaries")

with st.expander("Symptoms Per Day", expanded=True):
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

with st.expander("Energy & Mobility Scatter Plot"):
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

with st.expander("Top Symptoms Breakdown"):
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
                labels={"size": "Days observed", "Symptom": "Symptom"}
            )
            fig_sym_freq.update_layout(yaxis={"categoryorder": "total ascending"},
                                       margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_sym_freq, use_container_width=True)
        else:
            st.info("No symptom records in the selected range.")
    else:
        st.info("Symptoms column not available.")