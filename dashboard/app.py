"""
AtmosGrid — Satellite Intelligence for Air Quality and Atmospheric Hotspots.

Main Streamlit Application Entry Point with Top Segmented Mission Navigation.
ISRO-Inspired Saffron, Black & White Visual Theme.
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
    page_title="AtmosGrid — Geospatial Environmental Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Import Design Tokens & Custom Scoped CSS
from dashboard.theme import (
    ACCENT_SAFFRON,
    BG_SURFACE,
    BORDER_BASE,
    TEXT_MUTED,
    TEXT_WHITE,
    THEME_CSS,
)

st.markdown(THEME_CSS, unsafe_allow_html=True)


def main():
    # --- MISSION CONTROL HEADER ---
    st.markdown(
        f"""
        <div class="atmos-header">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div class="atmos-logo-icon">
                    🛰️
                </div>
                <div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 22px; font-weight: 800; letter-spacing: -0.02em; color: {TEXT_WHITE};">AtmosGrid</span>
                        <span class="atmos-badge">Operational Console</span>
                    </div>
                    <div style="font-size: 12px; color: {TEXT_MUTED}; margin-top: 2px;">
                        Satellite-Based Surface AQI Mapping & HCHO Hotspot Detection over India
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <div class="atmos-telemetry-pill">
                    <span style="color: {ACCENT_SAFFRON}; font-weight: 700;">●</span>
                    <span><b>Ground:</b> 6 CAAQMS (4 States)</span>
                </div>
                <div class="atmos-telemetry-pill">
                    <span style="color: {ACCENT_SAFFRON}; font-weight: 700;">●</span>
                    <span><b>Satellite:</b> Sentinel-5P HCHO (P90)</span>
                </div>
                <div class="atmos-telemetry-pill">
                    <span style="color: {ACCENT_SAFFRON}; font-weight: 700;">●</span>
                    <span><b>Coverage:</b> 36 States & UTs</span>
                </div>
                <div class="atmos-telemetry-pill">
                    <span style="color: #22C55E; font-weight: 700;">●</span>
                    <span><b>Model:</b> XGBoost (MAE 39.6)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- HORIZONTAL SEGMENTED MISSION NAVIGATION BAR ---
    page_options = [
        "Overview",
        "AQI Analytics",
        "HCHO Hotspot Explorer",
        "Data Quality & Methodology",
    ]

    page_selection = st.radio(
        label="Navigation",
        options=page_options,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="active_nav_page",
    )

    # --- FULL-WIDTH MAIN VIEW CANVAS ---
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
        f"""
        <div class="footer-text">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <b style="color: {TEXT_WHITE};">AtmosGrid Geospatial Intelligence Platform</b> · Developed for scientific environmental monitoring over India.
                </div>
                <div style="font-size: 11px; color: {TEXT_MUTED};">
                    Ground: CPCB / OpenAQ Open Data · Satellite: ESA / Copernicus Sentinel-5P · Boundaries: Survey of India
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
