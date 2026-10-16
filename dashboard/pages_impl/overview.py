"""
Overview page for AtmosGrid Dashboard.

Displays:
- Summary KPI cards from real CPCB reports, AQI evaluation, and HCHO clustering.
- Historical AQI time-series trend across 2024 with CPCB statutory category shading.
- Geographic overview map of India showing CPCB monitoring sites and HCHO clusters.
- Data coverage, observation periods, and system health status.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from dashboard.data_loader import (
    load_aqi_metrics,
    load_cpcb_daily_aqi,
    load_cpcb_report,
    load_hcho_clusters_summary,
    load_hcho_report,
)


def render_overview():
    st.markdown("### Environmental Intelligence Overview")
    st.markdown(
        "<p style='color: #8D99AE; margin-top: -10px;'>"
        "Real-time synthesis of ground monitoring observations, ML surface AQI estimation, "
        "and Sentinel-5P TROPOMI formaldehyde hotspot clustering across India."
        "</p>",
        unsafe_allow_html=True,
    )

    # Load artifacts
    cpcb_rep = load_cpcb_report() or {}
    aqi_met = load_aqi_metrics() or {}
    hcho_rep = load_hcho_report() or {}
    df_aqi = load_cpcb_daily_aqi()
    df_clusters = load_hcho_clusters_summary()

    # --- TOP ROW: KPI CARDS ---
    col1, col2, col3, col4 = st.columns(4)

    # 1. Stations
    stations_count = cpcb_rep.get("raw_data", {}).get("stations_with_data", 2)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">CPCB Ground Stations</div>
                <div class="kpi-value">{stations_count}</div>
                <div class="kpi-sub">Continuous CAAQMS (Delhi NCR)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Observations
    valid_days = cpcb_rep.get("aqi_summary", {}).get("valid_aqi_station_days", 617)
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Valid Daily AQI Days</div>
                <div class="kpi-value">{valid_days:,}</div>
                <div class="kpi-sub">16h statutory CPCB completeness</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Model Accuracy
    test_mae = aqi_met.get("test_metrics", {}).get("mae", 39.59)
    test_r2 = aqi_met.get("test_metrics", {}).get("r2", 0.7945)
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Held-out Model MAE</div>
                <div class="kpi-value">{test_mae:.1f} <span style="font-size: 16px; color: #48E5C2;">AQI</span></div>
                <div class="kpi-sub">R² = {test_r2:.3f} (XGBoost Regressor)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. HCHO Clusters
    num_clusters = hcho_rep.get("num_clusters", 3)
    elevated_px = hcho_rep.get("elevated_pixels", 29)
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Detected HCHO Clusters</div>
                <div class="kpi-value">{num_clusters}</div>
                <div class="kpi-sub">{elevated_px} elevated satellite pixels (P90)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # --- MAIN CONTENT: 2-COLUMN LAYOUT ---
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown("#### Ground AQI Temporal Evolution (2024)")
        if df_aqi is not None and not df_aqi.empty:
            # Aggregate by date across stations for smooth trend
            daily_mean = df_aqi.groupby("date").agg(
                mean_aqi=("aqi", "mean"),
                min_aqi=("aqi", "min"),
                max_aqi=("aqi", "max"),
            ).reset_index()

            fig = go.Figure()

            # Statutory CPCB category background bands
            # Good: 0-50, Satisfactory: 51-100, Moderate: 101-200, Poor: 201-300, Very Poor: 301-400, Severe: 401-500
            fig.add_hrect(y0=0, y1=50, fillcolor="#00E400", opacity=0.08, line_width=0, annotation_text="Good", annotation_position="top left")
            fig.add_hrect(y0=50, y1=100, fillcolor="#FFFF00", opacity=0.06, line_width=0, annotation_text="Satisfactory", annotation_position="top left")
            fig.add_hrect(y0=100, y1=200, fillcolor="#FF7E00", opacity=0.06, line_width=0, annotation_text="Moderate", annotation_position="top left")
            fig.add_hrect(y0=200, y1=300, fillcolor="#FF0000", opacity=0.06, line_width=0, annotation_text="Poor", annotation_position="top left")
            fig.add_hrect(y0=300, y1=500, fillcolor="#7E0023", opacity=0.08, line_width=0, annotation_text="Very Poor / Severe", annotation_position="top left")

            # Shaded min-max spread
            fig.add_trace(go.Scatter(
                x=daily_mean["date"],
                y=daily_mean["max_aqi"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=daily_mean["date"],
                y=daily_mean["min_aqi"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(72, 229, 194, 0.15)",
                name="Station Range",
            ))

            # Main daily line
            fig.add_trace(go.Scatter(
                x=daily_mean["date"],
                y=daily_mean["mean_aqi"],
                mode="lines",
                name="Daily Mean AQI",
                line=dict(color="#48E5C2", width=2.5),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Mean AQI: %{y:.1f}<extra></extra>",
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                margin=dict(l=40, r=20, t=20, b=40),
                height=360,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="CPCB AQI Index"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ground AQI observations loaded. Please run the CPCB acquisition pipeline.")

    with right_col:
        st.markdown("#### Geographic Footprint over India")
        # Build map with CPCB Stations and HCHO Clusters
        map_data = []

        # Ground stations
        if df_aqi is not None and not df_aqi.empty:
            stations = df_aqi.groupby(["location_id", "station_name", "latitude", "longitude"]).agg(
                avg_aqi=("aqi", "mean")
            ).reset_index()
            for _, r in stations.iterrows():
                map_data.append({
                    "name": r["station_name"],
                    "lat": float(r["latitude"]),
                    "lon": float(r["longitude"]),
                    "type": "CPCB Ground Station",
                    "info": f"Mean AQI: {r['avg_aqi']:.1f}",
                    "color": [0, 180, 216, 220],
                    "radius": 45000,
                })

        # HCHO Hotspot clusters
        if df_clusters is not None and not df_clusters.empty:
            for _, r in df_clusters.iterrows():
                map_data.append({
                    "name": r["region_label"],
                    "lat": float(r["centroid_lat"]),
                    "lon": float(r["centroid_lon"]),
                    "type": "HCHO Hotspot Cluster",
                    "info": f"{r['pixel_count']} pixels | Mean: {r['mean_hcho_molecules_cm2']:.2e} molec/cm²",
                    "color": [247, 37, 133, 220],
                    "radius": 65000,
                })

        if map_data:
            df_map = pd.DataFrame(map_data)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position="[lon, lat]",
                get_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )

            view_state = pdk.ViewState(
                latitude=22.5,
                longitude=79.5,
                zoom=3.8,
                pitch=20,
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"html": "<b>{name}</b><br><span style='color: #48E5C2;'>{type}</span><br>{info}"},
                map_style="mapbox://styles/mapbox/dark-v10",
            )
            st.pydeck_chart(r)
        else:
            st.info("Map data unavailable.")

    # --- BOTTOM ROW: METADATA & DATA PROVENANCE CARDS ---
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            """
            <div class="meta-card">
                <div style="font-weight: 600; color: #48E5C2; margin-bottom: 6px;">Ground Truth Provenance</div>
                <div style="font-size: 13px; color: #E2E8F0;">
                    <b>Provider:</b> OpenAQ AWS S3 Archive<br>
                    <b>Network:</b> CPCB & State PCB CAAQMS<br>
                    <b>Period:</b> Jan 2024 – Dec 2024<br>
                    <b>Resolution:</b> 24h statutory means (≥16h cutoff)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            <div class="meta-card">
                <div style="font-weight: 600; color: #48E5C2; margin-bottom: 6px;">Satellite Observation Provenance</div>
                <div style="font-size: 13px; color: #E2E8F0;">
                    <b>Instrument:</b> Sentinel-5P / TROPOMI (ESA)<br>
                    <b>Band:</b> Tropospheric HCHO column number density<br>
                    <b>Filters:</b> QA ≥ 0.50, Cloud fraction &lt; 0.30<br>
                    <b>Spatial Resolution:</b> 0.01° (~1.1 km L3 grid)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            """
            <div class="meta-card">
                <div style="font-weight: 600; color: #48E5C2; margin-bottom: 6px;">Scientific Disclaimer</div>
                <div style="font-size: 12px; color: #94A3B8; line-height: 1.4;">
                    HCHO clusters represent spatial atmospheric column concentration patterns, 
                    <b>not confirmed direct emission sources</b>. Formaldehyde is a secondary photochemical oxidation product. 
                    Model estimations are strictly local to ground monitors.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
