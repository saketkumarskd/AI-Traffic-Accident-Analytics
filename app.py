import streamlit as st
import pandas as pd
import joblib
import cv2
import tempfile
import time
import os
import shutil
import numpy as np
import plotly.graph_objects as go

from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Traffic & Accident Analytics",
    page_icon="🚦",
    layout="wide"
)


# =========================
# LOAD DEFAULT MODEL
# =========================
default_model = joblib.load("traffic_model.pkl")
default_scaler = joblib.load("scaler.pkl")


# =========================
# SESSION STATE
# =========================
if "active_model" not in st.session_state:
    st.session_state.active_model = default_model

if "active_scaler" not in st.session_state:
    st.session_state.active_scaler = default_scaler

if "active_dataset_name" not in st.session_state:
    st.session_state.active_dataset_name = "Default Trained Model"

if "uploaded_raw_df" not in st.session_state:
    st.session_state.uploaded_raw_df = None

if "uploaded_processed_df" not in st.session_state:
    st.session_state.uploaded_processed_df = None

if "uploaded_model_accuracy" not in st.session_state:
    st.session_state.uploaded_model_accuracy = None

if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False

if "traffic_dataset_ready" not in st.session_state:
    st.session_state.traffic_dataset_ready = False

if "manual_prediction_done" not in st.session_state:
    st.session_state.manual_prediction_done = False

if "manual_risk_level" not in st.session_state:
    st.session_state.manual_risk_level = None

if "manual_risk_score" not in st.session_state:
    st.session_state.manual_risk_score = None

if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False

if "video_detected_once" not in st.session_state:
    st.session_state.video_detected_once = False

if "video_last_frame" not in st.session_state:
    st.session_state.video_last_frame = None

if "video_risk_level" not in st.session_state:
    st.session_state.video_risk_level = None

if "video_risk_score" not in st.session_state:
    st.session_state.video_risk_score = None

if "video_vehicle_count" not in st.session_state:
    st.session_state.video_vehicle_count = None

if "video_avg_speed" not in st.session_state:
    st.session_state.video_avg_speed = None

if "saved_video_path" not in st.session_state:
    st.session_state.saved_video_path = None

if "saved_video_name" not in st.session_state:
    st.session_state.saved_video_name = None


# =========================
# CSS
# =========================
st.markdown("""
<style>
/* =========================
   GLOBAL LAYOUT FIX
========================= */
html {
    scroll-behavior: smooth;
    scroll-padding-top: 135px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(56,189,248,0.16), transparent 28%),
        radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 24%),
        linear-gradient(135deg,#020617,#0f172a,#111827);
    color: white;
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

.main .block-container {
    padding-top: 0.65rem !important;
    padding-bottom: 1.25rem !important;
    padding-left: clamp(0.35rem, 1.5vw, 1.2rem) !important;
    padding-right: clamp(0.35rem, 1.5vw, 1.2rem) !important;
    max-width: 100% !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] {
    display: none;
}

/* Remove excessive vertical gaps */
div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

.element-container {
    margin-bottom: 0.35rem !important;
}

/* =========================
   HERO SECTION
========================= */
.hero-box {
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.80));
    border: 1px solid rgba(56,189,248,0.26);
    border-radius: 18px;
    padding: 14px 10px 12px 10px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.24), inset 0 0 20px rgba(56,189,248,0.04);
    margin: 0 auto 6px auto;
    text-align: center;
    overflow: visible;
}

.title {
    font-size: clamp(21px, 3vw, 38px);
    line-height: 1.18;
    font-weight: 900;
    text-align: center;
    color: #38bdf8;
    text-shadow: 0 0 12px rgba(56,189,248,0.28);
    letter-spacing: 0.1px;
    margin: 0 auto 7px auto;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: normal;
}

.subtitle {
    text-align: center;
    font-size: clamp(12px, 1.2vw, 15px);
    line-height: 1.35;
    color: #cbd5e1;
    margin: 0 auto 10px auto;
    max-width: 100%;
    white-space: normal;
    overflow-wrap: anywhere;
}

.top-nav {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(108px, 1fr));
    gap: 6px;
    justify-content: center;
    align-items: stretch;
    margin: 8px auto 0 auto;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    overflow: visible;
}

.top-nav a {
    text-decoration: none;
    color: #e0f2fe;
    background: rgba(15,23,42,0.82);
    border: 1px solid rgba(56,189,248,0.24);
    padding: 8px 6px;
    border-radius: 12px;
    font-weight: 800;
    font-size: clamp(10px, 1vw, 12px);
    line-height: 1.15;
    box-shadow: 0 5px 14px rgba(0,0,0,0.18);
    transition: all 0.25s ease;
    white-space: normal;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 34px;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    overflow-wrap: anywhere;
}

.top-nav a:hover {
    background: linear-gradient(135deg,#0284c7,#2563eb);
    transform: translateY(-2px);
    box-shadow: 0 9px 18px rgba(14,165,233,0.28);
}

.top-nav a:focus,
.top-nav a:active {
    outline: 2px solid rgba(250,204,21,0.8);
    outline-offset: 2px;
    background: linear-gradient(135deg,#0369a1,#1d4ed8);
}

/* =========================
   SECTIONS
========================= */
.section {
    font-size: clamp(21px, 2vw, 29px);
    font-weight: 900;
    color: #f8fafc;
    margin-top: 22px;
    margin-bottom: 12px;
    padding: 12px 16px;
    border-radius: 16px;
    background: linear-gradient(90deg, rgba(30,41,59,0.94), rgba(15,23,42,0.58));
    border-left: 5px solid #38bdf8;
    box-shadow: 0 8px 20px rgba(0,0,0,0.20);
    scroll-margin-top: 135px;
    position: relative;
    z-index: 1;
}

.section:target {
    border-left: 7px solid #facc15;
    background: linear-gradient(90deg, rgba(14,165,233,0.36), rgba(15,23,42,0.78));
    box-shadow: 0 0 0 2px rgba(250,204,21,0.35), 0 12px 28px rgba(14,165,233,0.25);
    animation: sectionPulse 1.2s ease-in-out 1;
    padding-top: 14px;
    padding-bottom: 14px;
}

@keyframes sectionPulse {
    0% { transform: scale(1); }
    35% { transform: scale(1.012); }
    100% { transform: scale(1); }
}

/* =========================
   ALERT BOXES
========================= */
.red-box, .yellow-box, .green-box {
    padding: 16px;
    border-radius: 16px;
    text-align: center;
    font-weight: 900;
    transition: all 0.25s ease;
    box-shadow: 0 8px 22px rgba(0,0,0,0.24);
    line-height: 1.35;
}

.red-box:hover, .yellow-box:hover, .green-box:hover {
    transform: translateY(-3px) scale(1.005);
}

.red-box {
    background: linear-gradient(135deg,#7f1d1d,#dc2626);
    border: 2px solid #ef4444;
    font-size: 22px;
}

.yellow-box {
    background: linear-gradient(135deg,#78350f,#f59e0b);
    border: 2px solid #facc15;
    font-size: 22px;
}

.green-box {
    background: linear-gradient(135deg,#14532d,#16a34a);
    border: 2px solid #22c55e;
    font-size: 22px;
}

.info-card {
    background: rgba(15, 23, 42, 0.82);
    border: 1px solid rgba(148, 163, 184, 0.22);
    padding: 15px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 7px 20px rgba(0,0,0,0.20);
}

.info-card h3 {
    margin-top: 0 !important;
    margin-bottom: 0.4rem !important;
}

.info-card p {
    margin-bottom: 0.25rem !important;
}

/* =========================
   STATUS STRIP
========================= */
.status-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px;
    margin: 8px 0 4px 0;
    width: 100%;
    box-sizing: border-box;
}

.status-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.86));
    border: 1px solid rgba(56,189,248,0.18);
    padding: 10px 8px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 6px 16px rgba(0,0,0,0.20);
    min-height: 66px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
    overflow: hidden;
}

.status-card b {
    color: #38bdf8;
    font-size: clamp(12px, 1.2vw, 15px);
    line-height: 1.18;
    overflow-wrap: anywhere;
    word-break: normal;
}

.status-card span {
    display: block;
    color: #cbd5e1;
    font-size: clamp(10px, 1vw, 12px);
    line-height: 1.2;
    margin-top: 4px;
    overflow-wrap: anywhere;
}

/* =========================
   STREAMLIT COMPONENTS
========================= */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.92));
    border: 1px solid rgba(56,189,248,0.18);
    padding: 12px;
    border-radius: 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.22);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    transition: 0.22s ease;
    border-color: rgba(56,189,248,0.45);
}

.stButton > button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(56,189,248,0.35);
    background: linear-gradient(135deg,#0ea5e9,#2563eb);
    color: white;
    font-weight: 800;
    font-size: 16px;
    padding: 0.68rem 0.95rem;
    box-shadow: 0 8px 20px rgba(37,99,235,0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 12px 26px rgba(14,165,233,0.36);
    background: linear-gradient(135deg,#38bdf8,#1d4ed8);
}

.stFileUploader {
    background: rgba(15,23,42,0.72);
    padding: 8px;
    border-radius: 16px;
    border: 1px dashed rgba(56,189,248,0.28);
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

.stAlert {
    margin-top: 0.35rem !important;
    margin-bottom: 0.35rem !important;
}

h1, h2, h3, p {
    margin-top: 0.25rem !important;
}

hr {
    border: none;
    height: 1px;
    margin: 0.75rem 0 0.6rem 0;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.45), transparent);
}



/* =========================
   PREMIUM DASHBOARD HEADER ENHANCEMENT
========================= */
.hero-box {
    position: relative;
    overflow: hidden;
    border-top: 5px solid #38bdf8 !important;
    animation: headerFadeIn 0.9s ease both;
}

.hero-box::before {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -70px;
    top: -90px;
    border-radius: 50%;
    background: rgba(56,189,248,0.12);
    animation: floatingOrbOne 6s ease-in-out infinite;
}

.hero-box::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    left: -60px;
    bottom: -80px;
    border-radius: 50%;
    background: rgba(37,99,235,0.12);
    animation: floatingOrbTwo 7s ease-in-out infinite;
}

.title, .subtitle, .header-badge-row, .top-nav {
    position: relative;
    z-index: 2;
}

.title {
    animation: titleGlow 2.2s ease-in-out infinite alternate;
    transition: all 0.35s ease;
}

.title:hover {
    transform: scale(1.025);
    letter-spacing: 0.7px;
}

.header-badge-row {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px auto 10px auto;
    width: 100%;
}

.header-badge-chip {
    color: #e0f2fe;
    background: rgba(15,23,42,0.88);
    border: 1px solid rgba(56,189,248,0.24);
    padding: 9px 14px;
    border-radius: 999px;
    font-size: clamp(10px, 1vw, 12px);
    font-weight: 800;
    box-shadow: 0 6px 14px rgba(0,0,0,0.20);
    transition: all 0.28s ease;
}

.header-badge-chip:hover {
    background: linear-gradient(135deg,#38bdf8,#2563eb);
    color: #020617;
    transform: translateY(-4px) scale(1.045);
    box-shadow: 0 12px 26px rgba(56,189,248,0.32);
}

@keyframes titleGlow {
    from { text-shadow: 0 0 8px rgba(56,189,248,0.35); }
    to { text-shadow: 0 0 24px rgba(56,189,248,0.92); }
}

@keyframes floatingOrbOne {
    0% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(18px) scale(1.06); }
    100% { transform: translateY(0px) scale(1); }
}

@keyframes floatingOrbTwo {
    0% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-16px) scale(1.05); }
    100% { transform: translateY(0px) scale(1); }
}

@keyframes headerFadeIn {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 640px) {
    .header-badge-row {
        gap: 6px;
    }

    .header-badge-chip {
        padding: 7px 9px;
        font-size: 10px;
    }
}

/* =========================
   RESPONSIVE FIXES
========================= */
@media (max-width: 1000px) {
    .main .block-container {
        padding-left: 0.55rem !important;
        padding-right: 0.55rem !important;
    }

    .status-strip {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 7px;
    }

    .top-nav {
        grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
        gap: 5px;
    }

    .top-nav a {
        font-size: 10.5px;
        padding: 7px 5px;
        min-height: 32px;
    }
}

@media (max-width: 640px) {
    .main .block-container {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
        padding-top: 0.25rem !important;
    }

    .hero-box {
        padding: 10px 6px 9px 6px;
        border-radius: 14px;
    }

    .title {
        font-size: clamp(18px, 5.8vw, 25px);
        line-height: 1.2;
    }

    .subtitle {
        font-size: 11.2px;
        line-height: 1.3;
        margin-bottom: 8px;
    }

    .status-strip {
        grid-template-columns: 1fr 1fr;
        gap: 6px;
    }

    .status-card {
        min-height: 58px;
        padding: 8px 5px;
    }

    .status-card b {
        font-size: 11.2px;
    }

    .status-card span {
        font-size: 9.8px;
    }

    .top-nav {
        grid-template-columns: 1fr 1fr;
        gap: 5px;
        max-width: 100%;
        width: 100%;
    }

    .top-nav a {
        font-size: 10px;
        padding: 7px 4px;
        min-height: 30px;
        white-space: normal;
        min-width: 0;
        width: 100%;
        box-sizing: border-box;
    }

    .section {
        font-size: 18px;
        padding: 8px 9px;
    }

    .red-box, .yellow-box, .green-box {
        font-size: 16px;
        padding: 11px;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================
# RISK FUNCTIONS
# =========================
def calculate_risk(weather, vehicle_count, speed, time_value):
    score = 0

    if weather == "Rainy":
        score += 25

    if weather == "Foggy":
        score += 35

    if vehicle_count > 150:
        score += 25
    elif vehicle_count > 90:
        score += 15

    if speed > 90:
        score += 25
    elif speed > 60:
        score += 15

    if time_value in [8, 9, 10, 17, 18, 19]:
        score += 15

    score = min(score, 100)

    if score >= 70:
        return "HIGH", score

    if score >= 40:
        return "MEDIUM", score

    return "LOW", score


def signal_timing(risk_level):
    if risk_level == "HIGH":
        return 60, 25

    if risk_level == "MEDIUM":
        return 40, 50

    return 25, 45


def risk_box(risk_level, title_prefix=""):
    if risk_level == "HIGH":
        st.markdown(
            f'<div class="red-box">🚨 {title_prefix} HIGH ACCIDENT RISK</div>',
            unsafe_allow_html=True
        )

    elif risk_level == "MEDIUM":
        st.markdown(
            f'<div class="yellow-box">⚠️ {title_prefix} MEDIUM TRAFFIC RISK</div>',
            unsafe_allow_html=True
        )

    elif risk_level == "LOW":
        st.markdown(
            f'<div class="green-box">✅ {title_prefix} LOW / SAFE TRAFFIC FLOW</div>',
            unsafe_allow_html=True
        )

    else:
        st.info("Risk not checked yet.")


# =========================
# CHART FUNCTIONS
# =========================
def meter(title, value, max_value, color):
    safe_value = 0 if value is None else value

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=safe_value,
        title={"text": title, "font": {"size": 20, "color": "white"}},
        number={"font": {"size": 30, "color": "white"}},
        gauge={
            "axis": {
                "range": [0, max_value],
                "tickcolor": "white",
                "tickfont": {"color": "white"}
            },
            "bar": {"color": color},
            "bgcolor": "#0f172a",
            "borderwidth": 2,
            "bordercolor": "#475569",
            "steps": [
                {"range": [0, max_value * 0.33], "color": "#14532d"},
                {"range": [max_value * 0.33, max_value * 0.66], "color": "#78350f"},
                {"range": [max_value * 0.66, max_value], "color": "#7f1d1d"}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=260,
        margin=dict(l=10, r=10, t=45, b=10)
    )

    return fig


def traffic_3d_graph(graph_df=None):
    if graph_df is None:
        return None

    required_columns = ["Time", "Vehicle_Count", "Speed", "Accident"]

    for required_column in required_columns:
        if required_column not in graph_df.columns:
            return None

    chart_df = graph_df.copy()

    chart_df["Time"] = pd.to_numeric(chart_df["Time"], errors="coerce")
    chart_df["Vehicle_Count"] = pd.to_numeric(chart_df["Vehicle_Count"], errors="coerce")
    chart_df["Speed"] = pd.to_numeric(chart_df["Speed"], errors="coerce")
    chart_df["Accident"] = pd.to_numeric(chart_df["Accident"], errors="coerce")

    chart_df = chart_df.dropna(subset=["Time", "Vehicle_Count", "Speed", "Accident"])

    if chart_df.empty:
        return None

    chart_df["Risk_Intensity"] = (
        chart_df["Accident"] * 60
        + (chart_df["Vehicle_Count"] / chart_df["Vehicle_Count"].max()) * 25
        + (chart_df["Speed"] / chart_df["Speed"].max()) * 15
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=chart_df["Time"],
        y=chart_df["Vehicle_Count"],
        z=chart_df["Speed"],
        mode="markers",
        marker=dict(
            size=5,
            color=chart_df["Risk_Intensity"],
            colorscale="Turbo",
            opacity=0.85,
            colorbar=dict(title="Risk")
        ),
        text=[
            f"Time: {row.Time}<br>Vehicles: {row.Vehicle_Count}<br>Speed: {row.Speed}<br>Accident: {row.Accident}"
            for row in chart_df.itertuples()
        ],
        hoverinfo="text",
        name="Actual Dataset Points"
    ))

    fig.update_layout(
        title={
            "text": "Actual 3D Traffic Dataset Analytics",
            "font": {"color": "white", "size": 22}
        },
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=560,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(
                title="Time",
                backgroundcolor="#111827",
                gridcolor="#475569",
                zerolinecolor="#94a3b8",
                color="white"
            ),
            yaxis=dict(
                title="Vehicle Count",
                backgroundcolor="#111827",
                gridcolor="#475569",
                zerolinecolor="#94a3b8",
                color="white"
            ),
            zaxis=dict(
                title="Speed",
                backgroundcolor="#111827",
                gridcolor="#475569",
                zerolinecolor="#94a3b8",
                color="white"
            ),
            bgcolor="#020617"
        )
    )

    return fig


def plot_actual_accident_distribution(graph_df):
    accident_counts = graph_df["Accident"].value_counts().reset_index()
    accident_counts.columns = ["Accident", "Count"]

    accident_counts["Accident"] = accident_counts["Accident"].replace({
        0: "No Accident",
        1: "Accident"
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=accident_counts["Accident"],
        y=accident_counts["Count"],
        text=accident_counts["Count"],
        textposition="auto",
        name="Accident Count"
    ))

    fig.update_layout(
        title="Actual Accident Distribution",
        xaxis_title="Accident Class",
        yaxis_title="Count",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=380,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_actual_time_analysis(graph_df):
    time_df = graph_df.groupby("Time")["Accident"].sum().reset_index()
    time_df = time_df.sort_values("Time")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_df["Time"],
        y=time_df["Accident"],
        mode="lines+markers",
        name="Accidents"
    ))

    fig.update_layout(
        title="Actual Accidents by Time",
        xaxis_title="Hour",
        yaxis_title="Accident Count",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=380,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_actual_weather_analysis(graph_df):
    if "Weather" not in graph_df.columns:
        return None

    weather_df = graph_df.groupby("Weather")["Accident"].sum().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weather_df["Weather"],
        y=weather_df["Accident"],
        text=weather_df["Accident"],
        textposition="auto",
        name="Accidents"
    ))

    fig.update_layout(
        title="Actual Weather Impact on Accidents",
        xaxis_title="Weather",
        yaxis_title="Accident Count",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=380,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_actual_road_analysis(graph_df):
    if "Road_Type" not in graph_df.columns:
        return None

    road_df = graph_df.groupby("Road_Type")["Accident"].sum().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=road_df["Road_Type"],
        y=road_df["Accident"],
        text=road_df["Accident"],
        textposition="auto",
        name="Accidents"
    ))

    fig.update_layout(
        title="Actual Road Type Impact on Accidents",
        xaxis_title="Road Type",
        yaxis_title="Accident Count",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=380,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_actual_speed_vs_accident(graph_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=graph_df["Speed"],
        y=graph_df["Accident"],
        mode="markers",
        name="Speed vs Accident"
    ))

    fig.update_layout(
        title="Actual Speed vs Accident",
        xaxis_title="Speed",
        yaxis_title="Accident",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=380,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_generic_numeric_graph(graph_df, selected_numeric):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=graph_df[selected_numeric],
        mode="lines+markers",
        name=selected_numeric
    ))

    fig.update_layout(
        title=f"Trend of {selected_numeric}",
        xaxis_title="Record Index",
        yaxis_title=selected_numeric,
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_generic_category_graph(graph_df, selected_category):
    cat_df = graph_df[selected_category].value_counts().head(10).reset_index()
    cat_df.columns = [selected_category, "Count"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=cat_df[selected_category],
        y=cat_df["Count"],
        text=cat_df["Count"],
        textposition="auto"
    ))

    fig.update_layout(
        title=f"Distribution of {selected_category}",
        xaxis_title=selected_category,
        yaxis_title="Count",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "white"},
        height=400,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def prepare_graph_dataset():
    if st.session_state.dataset_uploaded and st.session_state.uploaded_raw_df is not None:
        graph_df = st.session_state.uploaded_raw_df.copy()
        source_name = "Uploaded Dataset"
    else:
        return None, "No Dataset Uploaded"

    graph_df.columns = graph_df.columns.astype(str).str.strip()

    for col in ["Time", "Vehicle_Count", "Speed", "Accident"]:
        if col in graph_df.columns:
            graph_df[col] = pd.to_numeric(graph_df[col], errors="coerce")

    if "Weather" in graph_df.columns:
        graph_df["Weather"] = graph_df["Weather"].astype(str).str.strip().str.title()

    if "Road_Type" in graph_df.columns:
        graph_df["Road_Type"] = graph_df["Road_Type"].astype(str).str.strip().str.title()

    return graph_df, source_name


# =========================
# DATASET PROCESSING
# =========================
def preprocess_uploaded_dataset(raw_df):
    required_columns = [
        "Weather",
        "Road_Type",
        "Time",
        "Vehicle_Count",
        "Speed",
        "Accident"
    ]

    missing_columns = []

    for column in required_columns:
        if column not in raw_df.columns:
            missing_columns.append(column)

    if len(missing_columns) > 0:
        return None, None, None, missing_columns

    df = raw_df.copy()

    df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])
    df["Road_Type"] = df["Road_Type"].fillna(df["Road_Type"].mode()[0])
    df["Time"] = df["Time"].fillna(df["Time"].median())
    df["Vehicle_Count"] = df["Vehicle_Count"].fillna(df["Vehicle_Count"].median())
    df["Speed"] = df["Speed"].fillna(df["Speed"].median())
    df["Accident"] = df["Accident"].fillna(df["Accident"].mode()[0])

    df.drop_duplicates(inplace=True)

    df["Weather"] = df["Weather"].astype(str).str.strip().str.title()
    df["Road_Type"] = df["Road_Type"].astype(str).str.strip().str.title()

    weather_map = {
        "Foggy": 0,
        "Rainy": 1,
        "Sunny": 2
    }

    road_map = {
        "City": 0,
        "Highway": 1,
        "Rural": 2
    }

    df = df[df["Weather"].isin(weather_map.keys())]
    df = df[df["Road_Type"].isin(road_map.keys())]

    df["Weather"] = df["Weather"].map(weather_map)
    df["Road_Type"] = df["Road_Type"].map(road_map)

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Vehicle_Count"] = pd.to_numeric(df["Vehicle_Count"], errors="coerce")
    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce")
    df["Accident"] = pd.to_numeric(df["Accident"], errors="coerce")

    df = df.dropna()

    df["Accident"] = df["Accident"].astype(int)

    df["Peak_Hour"] = df["Time"].apply(lambda x: 1 if int(x) in [8, 9, 10, 17, 18, 19] else 0)
    df["Day_Night"] = df["Time"].apply(lambda x: 1 if 6 <= int(x) <= 18 else 0)
    df["Traffic_Density"] = df["Vehicle_Count"] / (df["Speed"] + 1)

    scale_columns = [
        "Vehicle_Count",
        "Speed",
        "Traffic_Density"
    ]

    custom_scaler = StandardScaler()
    df[scale_columns] = custom_scaler.fit_transform(df[scale_columns])

    X = df.drop("Accident", axis=1)
    y = df["Accident"]

    if len(df) < 10:
        return None, None, None, ["Dataset rows are too low after cleaning. Minimum 10 rows needed."]

    if y.nunique() < 2:
        return None, None, None, ["Accident column must contain both 0 and 1 classes."]

    custom_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    custom_model.fit(X_train, y_train)

    predictions = custom_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return df, custom_model, custom_scaler, accuracy


# =========================
# VIDEO FILE STORAGE + SPEED
# =========================
def get_project_video_temp_folder():
    temp_video_folder = os.path.join(os.getcwd(), "temp_video_uploads")
    os.makedirs(temp_video_folder, exist_ok=True)
    return temp_video_folder


def clear_old_uploaded_video():
    if st.session_state.saved_video_path is not None:
        if os.path.exists(st.session_state.saved_video_path):
            try:
                os.remove(st.session_state.saved_video_path)
            except PermissionError:
                pass
            except OSError:
                pass

    st.session_state.saved_video_path = None
    st.session_state.saved_video_name = None
    st.session_state.video_last_frame = None
    st.session_state.video_detected_once = False
    st.session_state.video_vehicle_count = None
    st.session_state.video_avg_speed = None
    st.session_state.video_risk_level = None
    st.session_state.video_risk_score = None


def save_uploaded_video_safely(uploaded_video_file):
    temp_video_folder = get_project_video_temp_folder()

    original_name = uploaded_video_file.name.replace(" ", "_")
    file_extension = os.path.splitext(original_name)[1]

    if file_extension.lower() not in [".mp4", ".avi", ".mov"]:
        raise ValueError("Only MP4, AVI, and MOV videos are supported.")

    safe_video_name = "uploaded_traffic_video" + file_extension
    video_path = os.path.join(temp_video_folder, safe_video_name)

    if st.session_state.saved_video_name != uploaded_video_file.name:
        clear_old_uploaded_video()

        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except PermissionError:
                pass
            except OSError:
                pass

        uploaded_video_file.seek(0)

        with open(video_path, "wb") as output_video:
            output_video.write(uploaded_video_file.getbuffer())

        if not os.path.exists(video_path):
            raise FileNotFoundError("Video file could not be saved inside project folder.")

        if os.path.getsize(video_path) == 0:
            raise ValueError("Uploaded video file is empty or corrupted.")

        st.session_state.saved_video_path = video_path
        st.session_state.saved_video_name = uploaded_video_file.name

    return st.session_state.saved_video_path


def estimate_speed(box_area, vehicle_count):
    speed_value = int(120 - (box_area / 1000) - vehicle_count)
    speed_value = max(15, min(120, speed_value))
    return speed_value


# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero-box"><div class="title">🚦 AI Traffic & Accident Analytics Dashboard</div><div class="subtitle">Smart Traffic Monitoring • Accident Risk Prediction • Live Vehicle Detection • Real Dataset Analytics<br>Intelligent AI-powered transportation dashboard for smart city traffic management and accident prevention.</div><div class="header-badge-row"><span class="header-badge-chip">🤖 AI Prediction</span><span class="header-badge-chip">🚗 Vehicle Detection</span><span class="header-badge-chip">📊 Live Analytics</span><span class="header-badge-chip">🌐 3D Visualization</span></div><div class="top-nav"><a href="#upload-center-section">📂 Upload</a><a href="#controls-section">🎛️ Controls</a><a href="#video-detection-section">📹 Video Detection</a><a href="#prediction-section">🎯 Prediction</a><a href="#meters-section">📊 Meters</a><a href="#graph3d-section">🌐 3D Graph</a><a href="#eda-section">📈 EDA</a></div></div>
""", unsafe_allow_html=True)

st.markdown("---")


# =========================
# TOP STATUS OVERVIEW
# =========================
model_status = st.session_state.active_dataset_name
traffic_dataset_status = "Ready" if st.session_state.traffic_dataset_ready else "Not Uploaded"
video_status_text = "Detected" if st.session_state.video_detected_once else "Not Started"
prediction_status_text = "Done" if st.session_state.manual_prediction_done else "Pending"

st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-card"><b>{model_status}</b><span>Active ML Model</span></div>
        <div class="status-card"><b>{traffic_dataset_status}</b><span>Traffic Dataset</span></div>
        <div class="status-card"><b>{video_status_text}</b><span>Vehicle Detection</span></div>
        <div class="status-card"><b>{prediction_status_text}</b><span>Accident Prediction</span></div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# UPLOAD CENTER
# =========================
st.markdown('<div id="upload-center-section" class="section">📂 Upload Center</div>', unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    dataset_file = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"]
    )

with upload_col2:
    video_file = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov"]
    )

if dataset_file is not None:
    uploaded_df = pd.read_csv(dataset_file)
    uploaded_df.columns = uploaded_df.columns.astype(str).str.strip()

    st.session_state.dataset_uploaded = True
    st.session_state.uploaded_raw_df = uploaded_df

    st.success("Dataset uploaded successfully.")

    d1, d2, d3 = st.columns(3)

    d1.metric("Rows", uploaded_df.shape[0])
    d2.metric("Columns", uploaded_df.shape[1])
    d3.metric("Missing Values", int(uploaded_df.isnull().sum().sum()))

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(uploaded_df.head(20), use_container_width=True)

    processed_result, custom_model, custom_scaler, result_info = preprocess_uploaded_dataset(uploaded_df)

    if processed_result is not None:
        st.session_state.uploaded_processed_df = processed_result
        st.session_state.active_model = custom_model
        st.session_state.active_scaler = custom_scaler
        st.session_state.uploaded_model_accuracy = result_info
        st.session_state.active_dataset_name = "Uploaded Dataset Model"
        st.session_state.traffic_dataset_ready = True

        st.success("✅ Uploaded dataset successfully cleaned, preprocessed, and trained for prediction.")
        st.metric("Uploaded Dataset Model Accuracy", f"{round(result_info * 100, 2)}%")

    else:
        st.session_state.traffic_dataset_ready = False
        st.warning("Uploaded CSV preview shown, but it could not be used for accident model training.")
        st.write("Required columns:")
        st.write(["Weather", "Road_Type", "Time", "Vehicle_Count", "Speed", "Accident"])
        st.write("Issue:")
        st.write(result_info)

else:
    st.session_state.dataset_uploaded = False
    st.session_state.traffic_dataset_ready = False
    st.info("No dataset uploaded yet. Upload traffic dataset to enable responsive dataset analytics.")


# =========================
# ENVIRONMENT + SMART MANUAL CONTROLS
# =========================
st.markdown('<div id="controls-section" class="section">1️⃣ Smart Traffic Scenario Control Center</div>', unsafe_allow_html=True)

preset_col1, preset_col2 = st.columns([2, 1])

with preset_col1:
    scenario_mode = st.radio(
        "🚦 Quick Scenario Preset",
        ["Custom", "Safe Traffic", "Moderate Traffic", "High Risk Traffic"],
        horizontal=True,
        help="Select a ready-made traffic situation or choose Custom for manual control."
    )

if scenario_mode == "Safe Traffic":
    default_weather = "Sunny"
    default_road = "Highway"
    default_time = 11
    default_vehicle = 35
    default_speed = 45

elif scenario_mode == "Moderate Traffic":
    default_weather = "Rainy"
    default_road = "City"
    default_time = 18
    default_vehicle = 110
    default_speed = 72

elif scenario_mode == "High Risk Traffic":
    default_weather = "Foggy"
    default_road = "City"
    default_time = 19
    default_vehicle = 240
    default_speed = 110

else:
    default_weather = "Sunny"
    default_road = "Highway"
    default_time = 9
    default_vehicle = 120
    default_speed = 70

control_left, control_right = st.columns([2, 1])

with control_left:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🎛️ Traffic Condition Inputs")

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        weather = st.selectbox(
            "🌦 Weather Condition",
            ["Sunny", "Rainy", "Foggy"],
            index=["Sunny", "Rainy", "Foggy"].index(default_weather),
            help="Weather directly impacts accident probability. Foggy and rainy conditions increase risk."
        )

        road = st.selectbox(
            "🛣 Road Type",
            ["Highway", "City", "Rural"],
            index=["Highway", "City", "Rural"].index(default_road),
            help="Road type helps the ML model understand traffic environment."
        )

        time_value = st.slider(
            "⏰ Time of Day",
            0,
            23,
            default_time,
            help="Peak hours such as morning and evening traffic increase risk."
        )

    with input_col2:
        vehicle_count = st.slider(
            "🚗 Manual Vehicle Count",
            1,
            300,
            default_vehicle,
            help="Higher vehicle count means higher congestion and traffic density."
        )

        avg_speed = st.slider(
            "⚡ Manual Average Speed km/h",
            1,
            150,
            default_speed,
            help="Higher speed can increase accident severity and risk."
        )

    traffic_density_live = round(vehicle_count / max(avg_speed, 1), 2)
    live_preview_risk, live_preview_score = calculate_risk(weather, vehicle_count, avg_speed, time_value)

    st.markdown("#### 📊 Live Condition Intensity")
    st.progress(min(vehicle_count / 300, 1.0))

    preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)

    peak_status = "🔥 Peak Hour" if time_value in [8, 9, 10, 17, 18, 19] else "✅ Normal Hour"

    preview_col1.metric("Traffic Density", traffic_density_live)
    preview_col2.metric("Preview Risk", live_preview_risk)
    preview_col3.metric("Risk Score", live_preview_score)
    preview_col4.metric("Time Status", peak_status)

    st.markdown('</div>', unsafe_allow_html=True)

with control_right:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🧭 Demo Flow")
    st.write("✅ 1. Upload dataset/video")
    st.write("✅ 2. Select preset or custom controls")
    st.write("✅ 3. Run vehicle detection")
    st.write("✅ 4. Predict accident risk")
    st.write("✅ 5. View meters, 3D graph and EDA")
    st.markdown("---")
    st.write("**Current Scenario:**")
    st.write(f"Weather: {weather}")
    st.write(f"Road: {road}")
    st.write(f"Time: {time_value}:00")
    st.write(f"Vehicles: {vehicle_count}")
    st.write(f"Speed: {avg_speed} km/h")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# VIDEO DETECTION SECTION
# =========================
st.markdown('<div id="video-detection-section" class="section">2️⃣ Live Vehicle Detection & Video-Based Risk</div>', unsafe_allow_html=True)

st.warning(
    "Video risk and manual prediction are separated. Video detection updates only video status, and manual prediction updates only when you click Predict Accident Risk."
)

if video_file is not None:
    st.session_state.video_uploaded = True

    saved_video_path = None

    try:
        saved_video_path = save_uploaded_video_safely(video_file)
        st.success("Video uploaded safely inside project folder. Preview and detection output will appear in the same place.")

    except OSError as storage_error:
        saved_video_path = None
        st.error("Video save failed because disk/project storage is full or blocked. Delete old files from temp_video_uploads, free disk space, then upload again.")
        st.code(str(storage_error))

    except ValueError as video_value_error:
        saved_video_path = None
        st.error("Video upload failed because the uploaded file is not valid or supported.")
        st.code(str(video_value_error))

    except Exception as video_error:
        saved_video_path = None
        st.error("Video upload failed due to an unexpected issue.")
        st.code(str(video_error))

    video_col, status_col = st.columns([2, 1])

    with video_col:
        st.subheader("Uploaded / Detected Video Feed")
        video_placeholder = st.empty()

    with status_col:
        st.subheader("Live Video Status")
        vehicle_status = st.empty()
        speed_status = st.empty()
        risk_status = st.empty()
        signal_status = st.empty()
        alert_status = st.empty()

    if st.session_state.video_last_frame is None:
        if saved_video_path is not None:
            video_placeholder.video(saved_video_path)
        else:
            video_placeholder.warning("Video preview unavailable because file was not saved successfully.")
        vehicle_status.metric("Detected Vehicles", "Not started")
        speed_status.metric("Estimated Avg Speed", "Not started")
        risk_status.metric("Video Risk", "Not checked")
        signal_status.info("Click Start Continuous Vehicle Detection")

    if st.session_state.video_last_frame is not None:
        vgreen, vred = signal_timing(st.session_state.video_risk_level)

        video_placeholder.image(
            st.session_state.video_last_frame,
            channels="RGB",
            use_container_width=True,
            output_format="JPEG"
        )

        vehicle_status.metric("Detected Vehicles", st.session_state.video_vehicle_count)
        speed_status.metric("Estimated Avg Speed", f"{st.session_state.video_avg_speed} km/h")
        risk_status.metric("Saved Video Risk", st.session_state.video_risk_level)
        signal_status.info(f"Green {vgreen}s | Red {vred}s")
        risk_box(st.session_state.video_risk_level, "VIDEO:")

    if st.button("Start Continuous Vehicle Detection", use_container_width=True):
        yolo_model = YOLO("yolov8n.pt")
        if saved_video_path is None:
            st.error("Video file was not saved. Please free disk space and upload again.")
            st.stop()

        cap = cv2.VideoCapture(saved_video_path)

        vehicle_classes = [2, 3, 5, 7]
        frame_counter = 0
        max_frames = 600

        while cap.isOpened() and frame_counter < max_frames:
            success, frame = cap.read()

            if not success:
                break

            frame_counter += 1

            if frame_counter % 3 != 0:
                continue

            frame = cv2.resize(frame, (640, 360))
            results = yolo_model(frame, conf=0.25, verbose=False)

            live_count = 0
            speeds = []

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])

                    if class_id in vehicle_classes:
                        live_count += 1

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)

                        vehicle_speed = estimate_speed(area, live_count)
                        speeds.append(vehicle_speed)

                        if vehicle_speed > 90:
                            box_color = (0, 0, 255)
                            vehicle_risk_label = "HIGH"

                        elif vehicle_speed > 60:
                            box_color = (0, 255, 255)
                            vehicle_risk_label = "MEDIUM"

                        else:
                            box_color = (0, 255, 0)
                            vehicle_risk_label = "LOW"

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            box_color,
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Vehicle {vehicle_speed} km/h | {vehicle_risk_label}",
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            box_color,
                            2
                        )

            live_avg_speed = int(np.mean(speeds)) if speeds else 0

            video_risk, video_score = calculate_risk(
                weather,
                live_count,
                live_avg_speed,
                time_value
            )

            # Video detection result ko manual prediction state se separate rakha gaya hai.
            # Manual prediction sirf Predict Accident Risk button click karne par update hogi.

            if video_risk == "HIGH":
                overlay_color = (0, 0, 255)

            elif video_risk == "MEDIUM":
                overlay_color = (0, 255, 255)

            else:
                overlay_color = (0, 255, 0)

            live_green, live_red = signal_timing(video_risk)

            cv2.putText(frame, f"Vehicles: {live_count}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
            cv2.putText(frame, f"Avg Speed: {live_avg_speed} km/h", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)
            cv2.putText(frame, f"Video Risk: {video_risk}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, overlay_color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            st.session_state.video_last_frame = frame
            st.session_state.video_vehicle_count = live_count
            st.session_state.video_avg_speed = live_avg_speed
            st.session_state.video_risk_level = video_risk
            st.session_state.video_risk_score = video_score
            st.session_state.video_detected_once = True

            video_placeholder.image(frame, channels="RGB", use_container_width=True, output_format="JPEG")
            vehicle_status.metric("Detected Vehicles", live_count)
            speed_status.metric("Estimated Avg Speed", f"{live_avg_speed} km/h")
            risk_status.metric("Video Risk", video_risk)
            signal_status.info(f"Green {live_green}s | Red {live_red}s")

            if video_risk == "HIGH":
                alert_status.markdown('<div class="red-box">🚨 VIDEO: HIGH ACCIDENT RISK</div>', unsafe_allow_html=True)
            elif video_risk == "MEDIUM":
                alert_status.markdown('<div class="yellow-box">⚠️ VIDEO: MEDIUM TRAFFIC RISK</div>', unsafe_allow_html=True)
            else:
                alert_status.markdown('<div class="green-box">✅ VIDEO: LOW / SAFE TRAFFIC FLOW</div>', unsafe_allow_html=True)

            time.sleep(0.04)

        cap.release()
        st.success("Video detection completed and last result saved.")
        st.rerun()

else:
    st.session_state.video_uploaded = False
    st.info("No video uploaded yet. Upload traffic video to enable live vehicle detection.")


# =========================
# MANUAL ACCIDENT PREDICTION
# =========================
st.markdown('<div id="prediction-section" class="section">3️⃣ Manual Accident Risk Prediction</div>', unsafe_allow_html=True)

prediction_left, prediction_right = st.columns([2, 1])

with prediction_left:
    current_manual_risk, current_manual_score = calculate_risk(
        weather,
        vehicle_count,
        avg_speed,
        time_value
    )

    weather_map = {"Foggy": 0, "Rainy": 1, "Sunny": 2}
    road_map = {"City": 0, "Highway": 1, "Rural": 2}

    peak_hour = 1 if time_value in [8, 9, 10, 17, 18, 19] else 0
    day_night = 1 if 6 <= time_value <= 18 else 0
    traffic_density = vehicle_count / (avg_speed + 1)

    scale_input = pd.DataFrame([{
        "Vehicle_Count": vehicle_count,
        "Speed": avg_speed,
        "Traffic_Density": traffic_density
    }])

    scaled = st.session_state.active_scaler.transform(scale_input)

    input_data = pd.DataFrame([{
        "Weather": weather_map[weather],
        "Road_Type": road_map[road],
        "Time": time_value,
        "Vehicle_Count": scaled[0][0],
        "Speed": scaled[0][1],
        "Peak_Hour": peak_hour,
        "Day_Night": day_night,
        "Traffic_Density": scaled[0][2]
    }])

    if st.button("Predict Accident Risk", use_container_width=True):
        if not st.session_state.traffic_dataset_ready and not st.session_state.video_detected_once:
            st.warning("Upload a valid traffic accident CSV or run video detection first. No fake/manual-only prediction will be shown without real project input.")
            st.session_state.manual_prediction_done = False
            st.session_state.manual_risk_level = None
            st.session_state.manual_risk_score = None
        elif st.session_state.video_detected_once and not st.session_state.traffic_dataset_ready:
            st.info("Using latest video detection result as real prediction source.")
            st.session_state.manual_risk_level = st.session_state.video_risk_level
            st.session_state.manual_risk_score = st.session_state.video_risk_score
            st.session_state.manual_prediction_done = True
            st.rerun()
        else:
            ml_prediction = st.session_state.active_model.predict(input_data)[0]

            final_risk = current_manual_risk
            final_score = current_manual_score

            if ml_prediction == 1 and final_risk == "LOW":
                final_risk = "MEDIUM"
                final_score = max(final_score, 45)

            st.session_state.manual_risk_level = final_risk
            st.session_state.manual_risk_score = final_score
            st.session_state.manual_prediction_done = True
            st.rerun()

    if st.session_state.manual_prediction_done:
        risk_box(st.session_state.manual_risk_level, "MANUAL:")
    else:
        st.info("Click 'Predict Accident Risk' to generate manual prediction.")

with prediction_right:
    green, red = signal_timing(st.session_state.manual_risk_level)

    st.subheader("Smart Signal Timing")

    if st.session_state.manual_prediction_done:
        st.metric("Current Manual Risk", st.session_state.manual_risk_level)
        st.metric("Risk Score", st.session_state.manual_risk_score)
        st.metric("Green Signal", f"{green} sec")
        st.metric("Red Signal", f"{red} sec")
    else:
        st.metric("Current Manual Risk", "Not checked")
        st.metric("Risk Score", "Not checked")
        st.metric("Green Signal", "Not checked")
        st.metric("Red Signal", "Not checked")


# =========================
# METERS
# =========================
st.markdown('<div id="meters-section" class="section">4️⃣ AI Dashboard Meters</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

if st.session_state.video_detected_once:
    meter_risk_value = st.session_state.video_risk_score if st.session_state.video_risk_score is not None else 0
    meter_density_value = min(100, int(st.session_state.video_vehicle_count * 5)) if st.session_state.video_vehicle_count is not None else 0
    meter_speed_value = st.session_state.video_avg_speed if st.session_state.video_avg_speed is not None else 0

    m1.plotly_chart(
        meter("Video Accident Risk", meter_risk_value, 100, "#ef4444"),
        use_container_width=True,
        key="video_actual_risk_meter"
    )

    m2.plotly_chart(
        meter("Video Traffic Density", meter_density_value, 100, "#facc15"),
        use_container_width=True,
        key="video_actual_density_meter"
    )

    m3.plotly_chart(
        meter("Video Speed Level", meter_speed_value, 150, "#38bdf8"),
        use_container_width=True,
        key="video_actual_speed_meter"
    )

    st.success("Meters are working using latest video detection data.")

elif st.session_state.traffic_dataset_ready and st.session_state.manual_prediction_done:
    meter_risk_value = st.session_state.manual_risk_score if st.session_state.manual_risk_score is not None else 0
    meter_density_value = min(100, int(vehicle_count / 3))
    meter_speed_value = avg_speed

    m1.plotly_chart(
        meter("Dataset Accident Risk", meter_risk_value, 100, "#ef4444"),
        use_container_width=True,
        key="dataset_actual_risk_meter"
    )

    m2.plotly_chart(
        meter("Dataset Traffic Density", meter_density_value, 100, "#facc15"),
        use_container_width=True,
        key="dataset_actual_density_meter"
    )

    m3.plotly_chart(
        meter("Dataset Speed Level", meter_speed_value, 150, "#38bdf8"),
        use_container_width=True,
        key="dataset_actual_speed_meter"
    )

    st.success("Meters are working using uploaded dataset prediction data.")

else:
    m1.plotly_chart(
        meter("🔒 Accident Risk Locked", 0, 100, "#64748b"),
        use_container_width=True,
        key="locked_risk_meter"
    )

    m2.plotly_chart(
        meter("🔒 Traffic Density Locked", 0, 100, "#64748b"),
        use_container_width=True,
        key="locked_density_meter"
    )

    m3.plotly_chart(
        meter("🔒 Speed Level Locked", 0, 150, "#64748b"),
        use_container_width=True,
        key="locked_speed_meter"
    )

    st.warning("Meters are visible but locked. Upload a valid traffic dataset and click Predict Accident Risk, or run Vehicle Detection to activate them.")


# =========================
# 3D GRAPH SECTION
# =========================
st.markdown('<div id="graph3d-section" class="section">5️⃣ 3D Traffic Analytics</div>', unsafe_allow_html=True)

show_3d_graph = st.checkbox("Show 3D Traffic Risk Graph", value=True)

if show_3d_graph:
    if st.session_state.dataset_uploaded:
        graph_df_for_3d, graph_source_for_3d = prepare_graph_dataset()
        actual_3d_fig = traffic_3d_graph(graph_df_for_3d)

        if actual_3d_fig is not None:
            st.success(f"3D graph is using actual data from: {graph_source_for_3d}")
            st.plotly_chart(
                actual_3d_fig,
                use_container_width=True,
                key="traffic_3d_graph_actual_dataset"
            )
        else:
            st.warning("3D graph needs actual traffic columns: Time, Vehicle_Count, Speed, Accident.")
    elif st.session_state.video_detected_once:
        video_3d_df = pd.DataFrame([{
            "Time": time_value,
            "Vehicle_Count": st.session_state.video_vehicle_count,
            "Speed": st.session_state.video_avg_speed,
            "Accident": 1 if st.session_state.video_risk_level in ["HIGH", "MEDIUM"] else 0
        }])
        actual_3d_fig = traffic_3d_graph(video_3d_df)

        if actual_3d_fig is not None:
            st.success("3D graph is using latest video detection data.")
            st.plotly_chart(
                actual_3d_fig,
                use_container_width=True,
                key="traffic_3d_graph_video_data"
            )
        else:
            st.warning("Run video detection again to generate 3D analytics.")
    else:
        st.info("3D graph will appear only after uploading a dataset or running video detection.")
else:
    st.info("3D graph hidden. Tick checkbox to show it again.")


# =========================
# DATASET + RESPONSIVE ACTUAL EDA SECTION
# =========================
st.markdown('<div id="eda-section" class="section">6️⃣ Actual Dataset & Responsive EDA</div>', unsafe_allow_html=True)

if st.session_state.dataset_uploaded:

    show_dataset_eda = st.checkbox("Show Actual Dataset & Responsive Graphs", value=True)

    if show_dataset_eda:
        graph_df, graph_source_name = prepare_graph_dataset()

        if graph_df is not None:
            st.success(f"Graphs are using: {graph_source_name}")
            st.subheader("Actual Dataset Preview")
            st.dataframe(graph_df.head(30), use_container_width=True)

            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Rows", graph_df.shape[0])
            g2.metric("Columns", graph_df.shape[1])

            if "Accident" in graph_df.columns:
                accident_numeric = pd.to_numeric(graph_df["Accident"], errors="coerce").fillna(0)
                g3.metric("Accident Cases", int(accident_numeric.sum()))
                g4.metric("Non-Accident Cases", int((accident_numeric == 0).sum()))
            else:
                g3.metric("Accident Cases", "N/A")
                g4.metric("Non-Accident Cases", "N/A")

            numeric_columns = graph_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_columns = graph_df.select_dtypes(include=["object"]).columns.tolist()

            if "Accident" in graph_df.columns:
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.plotly_chart(plot_actual_accident_distribution(graph_df), use_container_width=True, key="actual_accident_distribution_uploaded")

                with chart_col2:
                    if "Time" in graph_df.columns:
                        st.plotly_chart(plot_actual_time_analysis(graph_df), use_container_width=True, key="actual_time_analysis_uploaded")
                    else:
                        st.warning("Time column not found.")

                chart_col3, chart_col4 = st.columns(2)

                with chart_col3:
                    weather_chart = plot_actual_weather_analysis(graph_df)
                    if weather_chart is not None:
                        st.plotly_chart(weather_chart, use_container_width=True, key="actual_weather_analysis_uploaded")
                    else:
                        st.warning("Weather column not found.")

                with chart_col4:
                    if "Speed" in graph_df.columns:
                        st.plotly_chart(plot_actual_speed_vs_accident(graph_df), use_container_width=True, key="actual_speed_vs_accident_uploaded")
                    else:
                        st.warning("Speed column not found.")

                road_chart = plot_actual_road_analysis(graph_df)

                if road_chart is not None:
                    st.plotly_chart(road_chart, use_container_width=True, key="actual_road_analysis_uploaded")

            else:
                st.warning("Uploaded dataset is not a traffic accident dataset because it does not contain the 'Accident' column. Showing generic dataset analysis instead.")

                if len(numeric_columns) > 0:
                    selected_numeric = st.selectbox("Select numeric column for graph", numeric_columns)
                    st.plotly_chart(plot_generic_numeric_graph(graph_df, selected_numeric), use_container_width=True, key="generic_numeric_graph_uploaded")

                if len(categorical_columns) > 0:
                    selected_category = st.selectbox("Select categorical column for distribution", categorical_columns)
                    st.plotly_chart(plot_generic_category_graph(graph_df, selected_category), use_container_width=True, key="generic_category_graph_uploaded")

                st.info("For accident prediction model training, upload CSV with exact columns: Weather, Road_Type, Time, Vehicle_Count, Speed, Accident.")

        else:
            st.info("No valid dataset found.")

    else:
        st.info("Dataset & EDA hidden.")

else:
    st.info("Dataset graphs will appear only after uploading a CSV dataset.")


# =========================
# FOOTER
# =========================
import streamlit.components.v1 as components

components.html("""
<style>
.footer-wrap{
    width:100%;
    box-sizing:border-box;
    padding:34px 22px;
    border-radius:28px;
    background:linear-gradient(135deg,#020617,#0f172a,#111827);
    border-top:5px solid #38bdf8;
    border-left:1px solid rgba(56,189,248,0.25);
    border-right:1px solid rgba(56,189,248,0.25);
    border-bottom:1px solid rgba(56,189,248,0.25);
    box-shadow:0 18px 45px rgba(0,0,0,0.45);
    font-family:Arial, sans-serif;
    color:white;
    overflow:hidden;
}

.footer-title{
    text-align:center;
    color:#38bdf8;
    font-size:36px;
    font-weight:900;
    margin-bottom:8px;
    animation:glow 2s infinite alternate;
}

.footer-sub{
    text-align:center;
    color:#cbd5e1;
    font-size:15px;
    line-height:1.7;
    margin-bottom:28px;
}

.footer-grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
    gap:20px;
}

.footer-card{
    min-height:210px;
    background:rgba(15,23,42,0.95);
    border:1px solid rgba(56,189,248,0.22);
    border-radius:22px;
    padding:22px;
    box-sizing:border-box;
    transition:all 0.35s ease;
    box-shadow:0 10px 25px rgba(0,0,0,0.25);
}

.footer-card:hover{
    transform:translateY(-8px) scale(1.02);
    border-color:#38bdf8;
    box-shadow:0 18px 35px rgba(56,189,248,0.22);
    background:linear-gradient(135deg,rgba(15,23,42,1),rgba(30,41,59,0.96));
}

.footer-card h3{
    color:#38bdf8;
    font-size:21px;
    margin-bottom:14px;
}

.footer-card p{
    color:#e2e8f0;
    line-height:1.9;
    font-size:14px;
    margin:0;
}

.footer-bottom{
    margin-top:30px;
    padding-top:18px;
    border-top:1px solid rgba(148,163,184,0.20);
    text-align:center;
    color:#94a3b8;
    font-size:13px;
    line-height:1.7;
}

.footer-badge{
    display:inline-block;
    margin-top:14px;
    padding:10px 18px;
    border-radius:999px;
    color:#38bdf8;
    background:rgba(14,165,233,0.12);
    border:1px solid rgba(56,189,248,0.25);
    font-weight:700;
}

@keyframes glow{
    from{text-shadow:0 0 6px rgba(56,189,248,0.35);}
    to{text-shadow:0 0 22px rgba(56,189,248,0.9);}
}

@media(max-width:700px){
    .footer-title{font-size:27px;}
    .footer-wrap{padding:24px 14px;}
    .footer-card{min-height:auto;}
}
</style>

<div class="footer-wrap">

    <div class="footer-title">
        🚦 AI Traffic & Accident Analytics
    </div>

    <div class="footer-sub">
        Smart Traffic Monitoring • AI Accident Prediction • Live Vehicle Detection • Real Dataset Analytics<br>
        A modern AI-powered dashboard for intelligent transportation and accident prevention research.
    </div>

    <div class="footer-grid">

        <div class="footer-card">
            <h3>📌 Core Features</h3>
            <p>
                ✅ Real Traffic Dataset Analytics<br>
                ✅ ML-Based Accident Prediction<br>
                ✅ YOLOv8 Vehicle Detection<br>
                ✅ Smart 3D Visualization
            </p>
        </div>

        <div class="footer-card">
            <h3>🛡 Smart Integrity</h3>
            <p>
                ✅ No Fake Prediction<br>
                ✅ Locked AI Meters<br>
                ✅ CSV Validation System<br>
                ✅ Safe Video Detection Flow
            </p>
        </div>

        <div class="footer-card">
            <h3>⚙ Technology Stack</h3>
            <p>
                ✅ Streamlit Dashboard<br>
                ✅ Random Forest ML Model<br>
                ✅ YOLOv8 Computer Vision<br>
                ✅ Plotly Interactive Charts
            </p>
        </div>

    </div>

    <div class="footer-bottom">
        Developed for smart traffic intelligence, accident prevention research, and AI-powered transportation systems.<br>
        ⚠️ Speed values are estimated for analytics purposes and may differ from actual radar/GPS systems.<br>
        <span class="footer-badge">IIT Patna Project Experience • Smart Traffic Analytics</span>
    </div>

</div>
""", height=620)