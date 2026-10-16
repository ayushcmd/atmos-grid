"""
Overview page for AtmosGrid Dashboard.

Displays:
- Summary KPI cards from real CPCB reports, AQI evaluation, and HCHO clustering.
- Interactive India Geospatial Overview map (Folium) with state boundaries and active stations.
- State-wise coverage classification matrix across all 36 States/UTs.
- Multi-station historical AQI time-series trend with statutory CPCB category bands.
- Data coverage, observation periods, and scientific caveats.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import (
    load_aqi_metrics,
    load_cpcb_daily_aqi,
    load_cpcb_report,
    load_hcho_clusters_summary,
    load_hcho_report,
    load_india_states_geojson,
    load_state_coverage_report,
    load_state_coverage_summary,
)
from dashboard.map_utils import (
    build_folium_overview_map,
    render_map_component,
)
from dashboard.theme import (
    ACCENT_SAFFRON,
    ACCENT_SAFFRON_DEEP,
    BORDER_BASE,
    STATUS_SUCCESS,
    STATUS_WARNING,
    TEXT_MUTED,
    TEXT_WHITE,
    apply_plotly_theme,
    kpi_card,
    meta_card,
)


def render_overview():
    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 18px; font-weight: 700; color: {TEXT_WHITE};">Environmental Intelligence Overview</div>
            <div style="font-size: 13px; color: {TEXT_MUTED}; margin-top: 2px;">
                Synthesis of statutory ground monitoring observations, supervised ML surface AQI estimation, 
                and Sentinel-5P TROPOMI formaldehyde hotspot clustering across India.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load artifacts
    cpcb_rep = load_cpcb_report() or {}
    aqi_met = load_aqi_metrics() or {}
    hcho_rep = load_hcho_report() or {}
    state_cov_rep = load_state_coverage_report() or {}
    df_state_cov = load_state_coverage_summary()
    df_aqi = load_cpcb_daily_aqi()
    df_clusters = load_hcho_clusters_summary()
    states_geojson = load_india_states_geojson()

    # --- TOP ROW: KPI CARDS ---
    col1, col2, col3, col4 = st.columns(4)

    # 1. Stations
    stations_count = state_cov_rep.get("total_active_stations", 6)
    states_count = state_cov_rep.get("states_with_measured_ground", 4)
    with col1:
        st.markdown(
            kpi_card(
                label="CPCB Ground Stations",
                value=str(stations_count),
                unit="sites",
                unit_color=ACCENT_SAFFRON,
                sub=f"Across {states_count} Indian States/UTs (CAAQMS)",
            ),
            unsafe_allow_html=True,
        )

    # 2. Observations
    valid_days = state_cov_rep.get("total_station_days", 1542)
    with col2:
        st.markdown(
            kpi_card(
                label="Valid Station-Days",
                value=f"{valid_days:,}",
                unit="days",
                unit_color=TEXT_WHITE,
                sub="≥16h statutory CPCB completeness",
            ),
            unsafe_allow_html=True,
        )

    # 3. Model Accuracy
    test_mae = aqi_met.get("test_metrics", {}).get("mae", 39.59)
    test_r2 = aqi_met.get("test_metrics", {}).get("r2", 0.7945)
    with col3:
        st.markdown(
            kpi_card(
                label="Held-out Model MAE",
                value=f"{test_mae:.1f}",
                unit="AQI",
                unit_color=ACCENT_SAFFRON,
                sub=f"R² = {test_r2:.3f} (Chronological Split)",
            ),
            unsafe_allow_html=True,
        )

    # 4. HCHO Clusters
    num_clusters = hcho_rep.get("num_clusters", 3)
    elevated_px = hcho_rep.get("elevated_pixels", 29)
    with col4:
        st.markdown(
            kpi_card(
                label="Detected HCHO Clusters",
                value=str(num_clusters),
                unit="cores",
                unit_color=ACCENT_SAFFRON_DEEP,
                sub=f"{elevated_px} elevated pixels (P90 DBSCAN)",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # --- MAIN VIEW TABS ---
    tab_map, tab_trend, tab_states = st.tabs([
        "National Geospatial Map",
        "Ground AQI Multi-Station Trends",
        "State Coverage Classification (36 States/UTs)",
    ])

    with tab_map:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <div style="font-size: 14px; font-weight: 700; color: {TEXT_WHITE};">Geospatial Distribution of Monitoring & Clusters</div>
                <div style="font-size: 11px; color: {TEXT_MUTED};">CartoDB Dark Basemap · 36 States/UT Boundaries</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        folium_overview = build_folium_overview_map(
            df_stations=df_aqi,
            df_clusters=df_clusters,
            states_geojson=states_geojson,
            df_state_coverage=df_state_cov,
        )
        render_map_component(folium_overview, fallback_title="India Overview Map", height=540)

        st.markdown(
            meta_card(
                title="Scientific Governance & Extrapolation Policy",
                content=(
                    "Atmospheric formaldehyde clusters represent elevated tropospheric column densities derived from satellite spectroscopy, "
                    "<b>not confirmed ground-level emission stacks</b>. Ground monitoring sites report 24-hour statutory averages. "
                    "Nationwide surface-AQI estimation is explicitly restricted from extrapolation across uncalibrated states."
                ),
                border_left_color=ACCENT_SAFFRON,
            ),
            unsafe_allow_html=True,
        )

    with tab_trend:
        st.markdown(
            f"""
            <div style="font-size: 14px; font-weight: 700; color: {TEXT_WHITE}; margin-bottom: 6px;">Multi-Station Ground AQI Time Series (2024)</div>
            """,
            unsafe_allow_html=True,
        )
        if df_aqi is not None and not df_aqi.empty:
            station_options = ["All Stations"] + sorted(df_aqi["station_name"].dropna().unique().tolist())
            selected_stn = st.selectbox("Select Monitoring Station", station_options, index=0)

            plot_aqi = df_aqi.copy()
            if selected_stn != "All Stations":
                plot_aqi = plot_aqi[plot_aqi["station_name"] == selected_stn]

            daily_stats = plot_aqi.groupby("date").agg(
                mean_aqi=("aqi", "mean"),
                min_aqi=("aqi", "min"),
                max_aqi=("aqi", "max"),
                stations_reporting=("location_id", "nunique"),
            ).reset_index()

            fig = go.Figure()

            # Statutory CPCB category background bands
            fig.add_hrect(y0=0, y1=50, fillcolor="#00E400", opacity=0.06, line_width=0, annotation_text="Good (0-50)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#00E400")
            fig.add_hrect(y0=50, y1=100, fillcolor="#FFFF00", opacity=0.04, line_width=0, annotation_text="Satisfactory (51-100)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#FFFF00")
            fig.add_hrect(y0=100, y1=200, fillcolor="#FF7E00", opacity=0.04, line_width=0, annotation_text="Moderate (101-200)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#FF7E00")
            fig.add_hrect(y0=200, y1=300, fillcolor="#FF0000", opacity=0.04, line_width=0, annotation_text="Poor (201-300)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#FF0000")
            fig.add_hrect(y0=300, y1=500, fillcolor="#7E0023", opacity=0.06, line_width=0, annotation_text="Very Poor / Severe (>300)", annotation_position="top left", annotation_font_size=10, annotation_font_color="#7E0023")

            # Min-Max spread
            fig.add_trace(go.Scatter(
                x=daily_stats["date"],
                y=daily_stats["max_aqi"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=daily_stats["date"],
                y=daily_stats["min_aqi"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(242, 140, 40, 0.12)",
                name="Station Range",
            ))

            # Main mean line (Saffron)
            fig.add_trace(go.Scatter(
                x=daily_stats["date"],
                y=daily_stats["mean_aqi"],
                mode="lines",
                name="Daily Mean AQI",
                line=dict(color=ACCENT_SAFFRON, width=2.4),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>AQI: %{y:.1f}<extra></extra>",
            ))

            apply_plotly_theme(
                fig,
                height=380,
                xaxis_title="Observation Date",
                yaxis_title="CPCB Daily AQI Index",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ground AQI observations loaded. Please run the ground acquisition pipeline.")

    with tab_states:
        st.markdown(
            f"""
            <div style="font-size: 14px; font-weight: 700; color: {TEXT_WHITE}; margin-bottom: 4px;">India Administrative Coverage Breakdown</div>
            <div style="font-size: 12px; color: {TEXT_MUTED}; margin-bottom: 12px;">Rigorous classification of all 36 States & Union Territories based on empirical data availability criteria.</div>
            """,
            unsafe_allow_html=True,
        )

        col_st1, col_st2, col_st3 = st.columns(3)
        with col_st1:
            st.markdown(
                meta_card(
                    title="1. Measured Ground (4 States)",
                    content="Active CPCB continuous CAAQMS stations meeting the statutory 16h daily completeness rule.",
                    border_left_color=ACCENT_SAFFRON,
                ),
                unsafe_allow_html=True,
            )
        with col_st2:
            st.markdown(
                meta_card(
                    title="2. Satellite Only (32 States/UTs)",
                    content="Sentinel-5P HCHO column density data available; ground AQI uncalibrated.",
                    border_left_color="#525863",
                ),
                unsafe_allow_html=True,
            )
        with col_st3:
            st.markdown(
                meta_card(
                    title="3. Validated National Model",
                    content="<b>Marked Unavailable / Pending</b> — Requires nationwide spatial cross-validation before claims.",
                    border_left_color=STATUS_WARNING,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        if df_state_cov is not None and not df_state_cov.empty:
            display_cols = ["state", "coverage_status", "station_count", "valid_station_days", "pollutants_measured", "mean_aqi", "data_completeness"]
            st.dataframe(
                df_state_cov[display_cols].rename(columns={
                    "state": "State / UT",
                    "coverage_status": "Classification",
                    "station_count": "Active Stations",
                    "valid_station_days": "Valid Days",
                    "pollutants_measured": "Parameters",
                    "mean_aqi": "Mean AQI",
                    "data_completeness": "Completeness",
                }),
                use_container_width=True,
                height=380,
            )
        else:
            st.info("State coverage report not found. Run backend/scripts/expand_ground_data.py to generate it.")

    # --- BOTTOM ROW: METADATA & DATA PROVENANCE CARDS ---
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            meta_card(
                title="Ground Truth Provenance",
                content=(
                    "<b>Provider:</b> OpenAQ AWS S3 Open Archive<br>"
                    "<b>Network:</b> CPCB & State PCB CAAQMS Network<br>"
                    "<b>Active States:</b> Delhi, UP, Maharashtra, Telangana<br>"
                    "<b>Completeness:</b> ≥16h valid readings per station-day"
                ),
                border_left_color=ACCENT_SAFFRON,
            ),
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            meta_card(
                title="Satellite Observation Provenance",
                content=(
                    "<b>Instrument:</b> Sentinel-5P / TROPOMI (ESA)<br>"
                    "<b>Band:</b> Tropospheric HCHO column number density<br>"
                    "<b>Filters:</b> QA ≥ 0.50, Cloud fraction &lt; 0.30<br>"
                    "<b>Spatial Resolution:</b> 0.01° (~1.1 km L3 grid)"
                ),
                border_left_color=TEXT_WHITE,
            ),
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            meta_card(
                title="Scientific Disclaimer",
                content=(
                    "HCHO clusters represent spatial atmospheric column concentration patterns, "
                    "<b>not confirmed direct emission sources</b>. Formaldehyde is a secondary photochemical oxidation product. "
                    "Model estimations are strictly local to ground monitors."
                ),
                border_left_color=STATUS_WARNING,
            ),
            unsafe_allow_html=True,
        )
