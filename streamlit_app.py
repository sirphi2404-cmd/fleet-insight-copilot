import os
import json
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from src.utils import load_report, detect_schema
from src.analytics import compute_facts_pack
from src.llm import generate_insights, chat_answer

load_dotenv()

st.set_page_config(page_title="Fleet Safety Insight Copilot", layout="wide")
st.title("Fleet Safety Insight Copilot (Local V1)")
st.caption("Upload a safety report (CSV/XLSX) → get insights + charts → chat follow-ups (grounded).")


# ---------- Helpers (define BEFORE use) ----------

def json_normalize_safe(rows):
    try:
        return pd.json_normalize(rows)
    except Exception:
        return pd.DataFrame(rows)

def _clean_field_name(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.split("(")[0].strip()
    s = re.sub(r"\s+", " ", s)
    return s

def resolve_field(df: pd.DataFrame, field: str) -> str | None:
    if not isinstance(field, str):
        return None

    raw = field.strip()
    cleaned = _clean_field_name(raw)

    if raw in df.columns:
        return raw
    if cleaned in df.columns:
        return cleaned

    aliases = {
        "driver": "entity",
        "driver name": "entity",
        "entity (driver)": "entity",
        "safety score": "primary_score",
        "primary_score": "primary_score",
        "total events": "total_events",
        "total_events": "total_events",
        "events per 100 exposure": "events_per_100_exposure",
        "events_per_100_exposure": "events_per_100_exposure",
        "distance travelled": "Distance Travelled",
        "distance": "Distance Travelled",
        "miles": "Distance Travelled",
        "rank": "rank",
        "cumulative share": "cumulative_share",
        "cumulative_share": "cumulative_share",
    }

    key = cleaned.lower()
    if key in aliases and aliases[key] in df.columns:
        return aliases[key]

    lower_map = {c.lower(): c for c in df.columns}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]
    if raw.lower() in lower_map:
        return lower_map[raw.lower()]

    return None

def choose_chart_df(facts_pack: dict, chart_spec: dict) -> pd.DataFrame | None:
    """
    Select the best chart dataset depending on chart needs.
    """
    chart_data = facts_pack.get("chart_data", {})

    # If chart needs Distance Travelled vs total_events, use exposure_vs_events
    x_raw = chart_spec.get("x", "")
    y_raw = chart_spec.get("y", "")

    x_l = _clean_field_name(str(x_raw)).lower()
    y_l = _clean_field_name(str(y_raw)).lower()

    if ("distance" in x_l or "miles" in x_l or "travelled" in x_l) and ("total_events" in y_l or "total events" in y_l):
        rows = chart_data.get("exposure_vs_events", [])
        return json_normalize_safe(rows) if rows else None

    # Pareto chart: if it asks for rank/cumulative share, use pareto curve
    if "pareto" in str(chart_spec.get("type", "")).lower() or ("cumulative" in x_l or "rank" in x_l or "cumulative" in y_l):
        rows = chart_data.get("pareto_curve_total_events", [])
        return json_normalize_safe(rows) if rows else None

    # Default: top risky entities
    rows = chart_data.get("top_risky_entities", [])
    return json_normalize_safe(rows) if rows else None

def render_chart(df: pd.DataFrame, chart_spec: dict):
    ctype = chart_spec.get("type")
    x_raw = chart_spec.get("x")
    y_raw = chart_spec.get("y")
    title = chart_spec.get("title", "Chart")

    x = resolve_field(df, x_raw) if x_raw else None
    y = resolve_field(df, y_raw) if y_raw else None

    if ctype == "histogram":
        if x is None:
            st.info(f"Chart skipped (missing x): {title} | x={x_raw}")
            return
        fig = px.histogram(df, x=x, title=title)
        st.plotly_chart(fig, use_container_width=True)
        return

    if x is None or y is None:
        st.info(f"Chart skipped (missing columns): {title} | x={x_raw}, y={y_raw}")
        return

    if ctype == "bar":
        fig = px.bar(df, x=x, y=y, title=title)
    elif ctype == "line":
        fig = px.line(df, x=x, y=y, title=title)
    elif ctype == "scatter":
        fig = px.scatter(df, x=x, y=y, title=title)
    elif ctype == "stacked_bar":
        series_raw = chart_spec.get("series")
        series = resolve_field(df, series_raw) if series_raw else None
        if series:
            fig = px.bar(df, x=x, y=y, color=series, title=title)
        else:
            fig = px.bar(df, x=x, y=y, title=title)
    elif ctype == "pareto":
        fig = px.line(df, x=x, y=y, title=title)
    else:
        st.info(f"Unknown chart type: {ctype}")
        return

    st.plotly_chart(fig, use_container_width=True)


# ---------- App state ----------

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY not found. Add it to .env and restart Streamlit.")

uploaded = st.file_uploader("Upload report (CSV/XLSX)", type=["csv", "xlsx"])

if "facts_pack" not in st.session_state:
    st.session_state.facts_pack = None
if "insights" not in st.session_state:
    st.session_state.insights = None
if "chat" not in st.session_state:
    st.session_state.chat = []


# ---------- Main flow ----------

if uploaded:
    with st.spinner("Parsing report..."):
        df = load_report(uploaded)
        schema = detect_schema(df)
        facts_pack = compute_facts_pack(df, schema)
        st.session_state.facts_pack = facts_pack

    st.success("Report parsed and facts computed.")

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Detected schema")
        st.json({
            "entity_col": schema.entity_col,
            "exposure_col": schema.exposure_col,
            "primary_score_col": schema.primary_score_col,
            "score_cols": schema.score_cols,
            "event_cols": schema.event_cols,
            "dim_cols": schema.dim_cols
        })

    with colB:
        st.subheader("Data preview")
        st.dataframe(df.head(30), use_container_width=True)

    st.divider()

    focus_q = st.text_input(
        "Optional: what do you want to analyze?",
        value="Who should we coach first and why?"
    )

    if st.button("Generate insights + charts (GPT-5.2)"):
        with st.spinner("Calling model for grounded insights..."):
            insights = generate_insights(st.session_state.facts_pack, user_question=focus_q)
            st.session_state.insights = insights

    if st.session_state.insights:
        insights = st.session_state.insights

        left, right = st.columns([1.1, 0.9])

        with left:
            st.subheader("Executive summary")
            for b in insights.get("executive_summary", []):
                st.write("• " + b)

            st.subheader("Key findings")
            for f in insights.get("key_findings", []):
                st.markdown(f"**{f.get('title','')}**")
                st.write(f.get("what_it_means",""))
                st.caption("Evidence:")
                for e in f.get("evidence", []):
                    st.write(f"- {e.get('metric')}: **{e.get('value')}** ({e.get('slice')})")
                st.write("")

            st.subheader("Recommended actions")
            for a in insights.get("recommended_actions", []):
                st.markdown(f"**{a.get('priority')} — {a.get('action')}**")
                st.write(f"Owner: {a.get('who')}")
                st.write(f"Expected: {a.get('expected_outcome')}")
                st.caption(a.get("why",""))
                st.write("")

            dq = insights.get("data_quality_notes", [])
            if dq:
                st.subheader("Data quality notes")
                for d in dq:
                    st.markdown(
                        f"- **Issue:** {d.get('issue')}  \n"
                        f"  Impact: {d.get('impact')}  \n"
                        f"  Fix: {d.get('suggested_fix')}"
                    )

        with right:
            st.subheader("Charts")
            st.caption("Charts automatically pick the right dataset (top risky / exposure scatter / pareto curve).")

            for ch in insights.get("charts", []):
                if ch.get("notes"):
                    st.caption(ch["notes"])

                chart_df = choose_chart_df(st.session_state.facts_pack, ch)
                if chart_df is None or chart_df.empty:
                    st.info(f"Chart skipped (no suitable dataset): {ch.get('title')}")
                    continue

                render_chart(chart_df, ch)

            st.subheader("Suggested follow-ups")
            for q in insights.get("follow_up_questions", []):
                st.write("• " + q)

    st.divider()

    st.subheader("Chat (grounded to this upload)")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    msg = st.chat_input("Ask a question about this report…")
    if msg:
        st.session_state.chat.append({"role": "user", "content": msg})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat_answer(st.session_state.facts_pack, st.session_state.chat, msg)
                st.write(answer)

        st.session_state.chat.append({"role": "assistant", "content": answer})
