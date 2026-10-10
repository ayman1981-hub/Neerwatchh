"""NEERWATCH Streamlit dashboard.

Reads the latest risk_snapshots and alerts from SQLite and re-renders every
3 seconds (streamlit-autorefresh; a sleep+rerun fallback keeps it working when
the package is absent).  Run it standalone with `streamlit run app.py` or via
run.py, which starts it as a subprocess next to the terminal simulation.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NEERWATCH", page_icon="💧", layout="wide")

# ------------------------------------------------------------------ theme --
# Classic-formal look: white canvas, navy headings with a double rule,
# serif display type (see .streamlit/config.toml for the light palette).
st.markdown(
    """
    <style>
    h1, h2, h3, h4 { color: #1F4E79 !important;
                     font-family: Georgia, "Times New Roman", serif !important; }
    header[data-testid="stHeader"] { background: transparent; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #B7C4D6 !important; background: #F7F9FC !important; }
    .nw-rule { border-bottom: 3px double #1F4E79; padding-bottom: .35rem;
               margin-bottom: .5rem; }
    .nw-sub  { color: #5B6B7F; font-style: italic; font-size: .92rem; }
    [data-testid="stMetricValue"] { font-family: Georgia, serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    from streamlit_autorefresh import st_autorefresh

    st_autorefresh(interval=3000, key="neerwatch_autorefresh")
    AUTO_REFRESH = True
except Exception:
    AUTO_REFRESH = False  # fallback: sleep + rerun at the bottom of the script

from src import alert as alert_mod  # noqa: E402
from src import db  # noqa: E402
from src import pipeline  # noqa: E402

RED, AMBER, GREEN = "#e53935", "#fb8c00", "#43a047"


def risk_color(score: float, has_alert: bool) -> str:
    if has_alert or score >= alert_mod.ANOMALY_THRESHOLD:
        return RED
    if score >= 0.50:
        return AMBER
    return GREEN


def load_data():
    conn = db.connect()
    try:
        snapshots = db.latest_snapshots(conn)
        alerts = db.get_active_alerts(conn)
        since = (datetime.now() - timedelta(hours=48)).isoformat(timespec="seconds")
        feed = db.recent_reports_feed(conn, since)
        history_wards = (
            snapshots.dropna(subset=["ts"])["ward_id"].astype(int).tolist()
            if not snapshots.empty and "ts" in snapshots
            else []
        )
    finally:
        conn.close()
    return snapshots, alerts, feed, history_wards


snapshots, alerts, feed, history_wards = load_data()

# ---------------------------------------------------------------- header ----
st.markdown(
    '<div class="nw-rule"><h2 style="margin:0">💧 NEERWATCH — Street-level '
    'Early Warning for Unsafe Drinking Water</h2>'
    '<div class="nw-sub">Prototype command dashboard · ward-level risk · '
    'anomaly + cluster rule engine</div></div>',
    unsafe_allow_html=True,
)
h1, h2 = st.columns([4, 1])
with h2:
    st.markdown(f"**{datetime.now():%Y-%m-%d %H:%M:%S}**")
    st.caption("auto-refresh: 3s" if AUTO_REFRESH else "auto-refresh: fallback loop")

scored = snapshots.dropna(subset=["score"])
n_reports = int(feed.shape[0]) if not feed.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("🚨 Active alerts", len(alerts))
k2.metric("📍 Wards monitored", len(snapshots))
k3.metric("📩 Reports (48h)", n_reports)
k4.metric(
    "📈 Max risk score",
    f"{scored['score'].max():.2f}" if not scored.empty else "—",
)

# ------------------------------------------------- data provenance note -----
with st.container(border=True):
    p1, p2, p3 = st.columns(3)
    p1.markdown("**✅ Real:** Open-Meteo rainfall · ward geometry · trained ML models")
    p2.markdown("**🧪 Simulated:** citizen reports · PWD leak log · outbreak scenarios")
    p3.markdown("**🔲 Out of scope (future):** Telegram / LLM / voice · resident "
                "alerts · login · storm mode")

# ------------------------------------------------ live report intake --------
with st.container(border=True):
    st.markdown("#### 📡 Live citizen report intake")
    st.caption("Type a message exactly as a resident would send it (Tamil or English).")
    with st.form("intake", clear_on_submit=True):
        intake_cols = st.columns([4, 1])
        with intake_cols[0]:
            text_in = st.text_input(
                "Report text",
                placeholder="உதா: வார்டு 5 தண்ணீரில் நாற்றம், 3 பேருக்கு வயிற்றுப்போக்கு",
                label_visibility="collapsed",
            )
        with intake_cols[1]:
            submitted = st.form_submit_button("Submit report", use_container_width=True)
    if submitted and text_in.strip():
        stages = pipeline.process_report(text_in.strip(), source="live")
        ex = stages.get("extract") or {}
        if stages.get("note"):
            st.session_state["last_intake"] = (
                "warning",
                f"Could not find a ward number in the message — {stages['note']}",
            )
        else:
            msg = (f"Ingested → **ward {ex.get('ward')}** · symptom **{ex.get('symptom')}** "
                   f"· **{ex.get('cases')}** case(s)")
            if (stages.get("alert") or {}).get("fired"):
                st.session_state["last_intake"] = (
                    "error", msg + " — **alert raised for this ward**")
            else:
                st.session_state["last_intake"] = (
                    "success", msg + " — no alert threshold crossed")
    # keep the confirmation on screen across the 3s auto-refresh reruns
    _last = st.session_state.get("last_intake")
    if _last:
        getattr(st, _last[0])(_last[1])

# ------------------------------------------------------------ alert cards ---
st.markdown("---")
if alerts.empty:
    st.success("✅ No active alerts — all wards below the anomaly threshold.")
else:
    for row in alerts.itertuples():
        reasons = row.reasons if isinstance(row.reasons, list) else json.loads(row.reasons)
        actions = row.actions if isinstance(row.actions, list) else json.loads(row.actions)
        try:
            mins = int((datetime.now() - datetime.fromisoformat(row.ts)).total_seconds() // 60)
            since = f"{mins} min ago" if mins >= 0 else "just now"
        except Exception:
            since = row.ts
        st.error(
            f"🚨 **ACTIVE ALERT — {row.name}**  ·  severity: **{row.severity.upper()}**  "
            f"·  anomaly score **{row.score:.2f}**  ·  raised {since}"
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"**Why:** {alert_mod.reasons_line(reasons)}")
        with c2:
            st.markdown("**Suggested actions:**")
            for act in actions:
                st.markdown(f"- {act}")

st.markdown("---")

# -------------------------------------------------------------- map ---------
map_col, side_col = st.columns([3, 2])

with map_col:
    st.markdown("### 🗺️ Ward risk map")
    if scored.empty:
        st.info("Waiting for the first risk snapshots… start `python run.py --replay`.")
    else:
        import folium
        from streamlit.components import v1 as components

        center = [snapshots["lat"].mean(), snapshots["lng"].mean()]
        fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")
        alert_wards = set(alerts["ward_id"].tolist()) if not alerts.empty else set()

        for row in snapshots.itertuples():
            score = 0.0 if pd.isna(getattr(row, "score")) else float(row.score)
            has_alert = int(row.ward_id) in alert_wards
            color = risk_color(score, has_alert)
            reports = 0 if pd.isna(getattr(row, "reports_48h")) else int(row.reports_48h)
            popup = (
                f"<b>{row.name}</b><br>score: {score:.2f}<br>"
                f"reports 48h: {reports}<br>"
                f"rain 48h: {row.rain_mm if not pd.isna(row.rain_mm) else 0} mm<br>"
                f"leak: {'yes' if not pd.isna(row.has_leak) and int(row.has_leak) else 'no'}"
            )
            folium.CircleMarker(
                location=[row.lat, row.lng],
                radius=7 + min(reports, 6) * 2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                weight=2,
                popup=folium.Popup(popup, max_width=260),
                tooltip=f"{row.name} · score {score:.2f}",
            ).add_to(fmap)

        components.html(fmap._repr_html_(), height=520)

        st.caption(
            "🔴 anomaly ≥ 0.70 or active alert   ·   🟠 ≥ 0.50   ·   🟢 below"
        )

with side_col:
    st.markdown("### 📊 Ward risk table")
    if scored.empty:
        st.caption("No snapshots yet.")
    else:
        table = scored.copy()
        table["score"] = table["score"].astype(float).round(2)
        cols = ["ward_id", "name", "score", "reports_48h", "cases_48h",
                "cluster_size", "rain_mm", "has_leak"]
        table = table[[c for c in cols if c in table.columns]].rename(columns={
            "ward_id": "Ward", "name": "Name", "score": "Score",
            "reports_48h": "Reports", "cases_48h": "Cases",
            "cluster_size": "Cluster", "rain_mm": "Rain(mm)", "has_leak": "Leak",
        })
        st.dataframe(table.sort_values("Score", ascending=False), use_container_width=True,
                     hide_index=True)

        if history_wards:
            st.markdown("### 📈 Score timeline")
            ward_ids = sorted(set(history_wards))
            names = snapshots.set_index("ward_id")["name"].to_dict()
            pick = st.selectbox(
                "Ward", ward_ids,
                format_func=lambda w: f"{w} — {names.get(w, '')}",
                key="timeline_ward",
            )
            hist = db.snapshot_history(db.connect(), int(pick))
            if not hist.empty:
                hist = hist.set_index("ts")[["score", "anomaly_score"]].round(2)
                st.line_chart(hist, height=220)

# ---------------------------------------------------------- report feed -----
st.markdown("### 📥 Live report feed (48h)")
if feed.empty:
    st.info("No reports ingested yet.")
else:
    feed_view = feed.copy()
    feed_view.columns = ["Time", "Ward", "Symptom", "Cases", "Message", "Source"]
    st.dataframe(feed_view, use_container_width=True, hide_index=True, height=300)

st.caption(
    "Pipeline: sentence-transformers + Logistic Regression (symptoms) → "
    "DBSCAN (48h clusters) → Isolation Forest (anomaly) → rule engine "
    "(anomaly > 0.70 AND ≥ 3 reports/48h)."
)

if not AUTO_REFRESH:
    time.sleep(3)
    st.rerun()
