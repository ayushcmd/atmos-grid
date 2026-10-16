"""
AtmosGrid — Satellite Intelligence for Air Quality and Atmospheric Hotspots.

Main Streamlit Application Entry Point with Right-Hand Navigation Rail.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (parent directory of dashboard) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Configure wide layout with collapsed sidebar for maximum visualization canvas
st.set_page_config(
    page_title="AtmosGrid — Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Theme CSS (Dark Navy & Slate with Cyan/Teal Accents & Right Rail Styling)
CUSTOM_CSS = """
<style>
    /* Dark Navy Main Background */
    .stApp {
        background-color: #0B132B;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top Header Bar */
    .atmos-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        background: #0F172A;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 20px;
    }

    /* KPI Cards */
    .kpi-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(72, 229, 194, 0.4);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748B;
        margin-top: 4px;
    }

    /* Information / Metadata Card */
    .meta-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 14px 16px;
    }

    /* Right Navigation Styling */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    div[data-testid="stRadio"] label {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 10px 14px;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
        color: #CBD5E1 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: rgba(72, 229, 194, 0.5);
        background: rgba(72, 229, 194, 0.06);
    }
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] input:checked + div {
        color: #48E5C2 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: #94A3B8;
        font-weight: 500;
        font-size: 13px;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(72, 229, 194, 0.08) !important;
        color: #48E5C2 !important;
        border-bottom: 2px solid #48E5C2 !important;
    }

    /* Footer */
    .footer-text {
        font-size: 12px;
        color: #64748B;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 15px;
        margin-top: 40px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    # --- BRANDED TOP HEADER BANNER ---
    st.markdown(
        """
        <div class="atmos-header">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="background: linear-gradient(135deg, #48E5C2 0%, #00B4D8 100%); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                    🌍
                </div>
                <div>
                    <div style="display: flex; align-items: baseline; gap: 8px;">
                        <span style="font-size: 22px; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF;">AtmosGrid</span>
                        <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #48E5C2; font-weight: 700; background: rgba(72, 229, 194, 0.12); padding: 2px 8px; border-radius: 4px;">Phase 2A Live</span>
                    </div>
                    <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">
                        Satellite Intelligence for Air Quality and Atmospheric Hotspots over India.
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 16px; font-size: 11px; color: #94A3B8;">
                <div>🏢 <b>Ground:</b> 6 CAAQMS Stations (4 States)</div>
                <div>🛰️ <b>Satellite:</b> Sentinel-5P HCHO (P90)</div>
                <div>🗺️ <b>Boundaries:</b> 36 States & UTs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- PRIMARY CANVAS WITH RIGHT-HAND NAVIGATION RAIL ---
    col_main, col_nav = st.columns([0.83, 0.17], gap="medium")

    # --- RIGHT NAVIGATION PANEL ---
    with col_nav:
        st.markdown(
            """
            <div style="font-size: 11px; font-weight: 700; color: #48E5C2; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px;">
                🧭 Navigation Rail
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_options = [
            "Overview",
            "AQI Analytics",
            "HCHO Hotspot Explorer",
            "Data Quality & Methodology",
        ]

        page_selection = st.radio(
            label="Dashboard Pages",
            options=page_options,
            index=0,
            label_visibility="collapsed",
            key="active_nav_page",
        )

        st.markdown(
            """
            <div class="kpi-card" style="margin-top: 18px; padding: 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 8px;">System Telemetry</div>
                <div style="font-size: 11px; color: #CBD5E1; line-height: 1.6;">
                    <b>Model:</b> XGBoost Regressor (3.4)<br>
                    <b>Spatial:</b> Haversine DBSCAN (ε=50km)<br>
                    <b>Ground:</b> OpenAQ S3 / CPCB<br>
                    <b>Satellite:</b> Sentinel-5P TROPOMI<br>
                    <b>Boundaries:</b> Survey of India 36 UTs
                </div>
            </div>

            <div class="meta-card" style="margin-top: 14px; font-size: 11px; color: #64748B; line-height: 1.4;">
                <b style="color: #94A3B8;">Scientific Governance:</b><br>
                Predictions are station-localized. Nationwide spatial extrapolation is restricted pending multi-regional validation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- MAIN VIEW CANVAS ---
    with col_main:
        if page_selection == "Overview":
            from dashboard.pages_impl.overview import render_overview
            render_overview()
        elif page_selection == "AQI Analytics":
            from dashboard.pages_impl.aqi_analytics import render_aqi_analytics
            render_aqi_analytics()
        elif page_selection == "HCHO Hotspot Explorer":
            from dashboard.pages_impl.hcho_explorer import render_hcho_explorer
            render_hcho_explorer()
        elif page_selection == "Data Quality & Methodology":
            from dashboard.pages_impl.data_quality import render_data_quality
            render_data_quality()

    # --- FOOTER ---
    st.markdown(
        """
        <div class="footer-text">
            <b>AtmosGrid Geospatial Intelligence Platform</b> · Developed for scientific environmental monitoring over India. 
            Ground observations licensed via CPCB / OpenAQ Open Data. Satellite data courtesy ESA / Copernicus Sentinel-5P.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
