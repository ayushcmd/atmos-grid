"""
HCHO Hotspot Explorer page for AtmosGrid Dashboard.

Displays:
- Interactive geographic Folium map with working CartoDB dark basemap, Indian states,
  and Sentinel-5P TROPOMI HCHO cluster footprints.
- Cluster breakdown with coordinates, pixel count, mean HCHO, and peak concentrations.
- DBSCAN clustering parameters (eps=50km, min_samples=5, threshold=P90).
- Formaldehyde column density distribution histogram.
- Scientific limitations and emission-source caveats.
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
    load_hcho_benchmark,
    load_hcho_clusters_summary,
    load_hcho_hotspots_geojson,
    load_hcho_report,
    load_india_states_geojson,
)
from dashboard.map_utils import (
    build_folium_hcho_map,
    render_map_component,
)
from dashboard.theme import (
    ACCENT_SAFFRON,
    ACCENT_SAFFRON_DEEP,
    BORDER_BASE,
    STATUS_WARNING,
    TEXT_MUTED,
    TEXT_OFFWHITE,
    TEXT_WHITE,
    apply_plotly_theme,
    kpi_card,
    meta_card,
)


def render_hcho_explorer():
    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 18px; font-weight: 700; color: {TEXT_WHITE};">Sentinel-5P HCHO Hotspot Explorer</div>
            <div style="font-size: 13px; color: {TEXT_MUTED}; margin-top: 2px;">
                Spatial density-based clustering (Haversine DBSCAN) applied to TROPOMI formaldehyde tropospheric column measurements over India.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load artifacts
    df_clusters = load_hcho_clusters_summary()
    geojson_data = load_hcho_hotspots_geojson()
    report = load_hcho_report() or {}
    df_raw_pts = load_hcho_benchmark()
    states_geojson = load_india_states_geojson()

    params = report.get("clustering_parameters", {})

    # --- TOP ROW: KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                label="Identified Hotspots",
                value=str(report.get("num_clusters", 3)),
                unit="clusters",
                unit_color=ACCENT_SAFFRON,
                sub="Haversine DBSCAN Spatial Clustering",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        thresh_val = params.get("threshold_value_molecules_cm2", 1.70e16)
        st.markdown(
            kpi_card(
                label="DBSCAN Threshold",
                value=f"P{params.get('threshold_percentile', 90.0):.0f}",
                unit="cutoff",
                unit_color=TEXT_WHITE,
                sub=f"≥ {thresh_val:.2e} molec/cm²",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card(
                label="Neighborhood Radius",
                value=f"{params.get('eps_km', 50.0):.0f}",
                unit="km",
                unit_color=ACCENT_SAFFRON,
                sub="Haversine metric (min_samples=5)",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card(
                label="Clustered Footprint",
                value=str(report.get("clustered_pixels", 29)),
                unit="pixels",
                unit_color=ACCENT_SAFFRON_DEEP,
                sub=f"Across {report.get('num_clusters', 3)} distinct spatial cores",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # --- MAP + DETAILS SECTION ---
    col_map, col_details = st.columns([1.25, 0.75])

    with col_map:
        st.markdown(
            f"""
            <div style="font-size: 14px; font-weight: 700; color: {TEXT_WHITE}; margin-bottom: 4px;">Spatial Hotspot Cluster Distribution</div>
            <div style="font-size: 11px; color: {TEXT_MUTED}; margin-bottom: 8px;">Interactive basemap with Indian administrative boundaries and HCHO cluster footprints.</div>
            """,
            unsafe_allow_html=True,
        )

        if df_clusters is not None and not df_clusters.empty:
            folium_hcho = build_folium_hcho_map(
                df_clusters=df_clusters,
                df_raw_pts=df_raw_pts,
                states_geojson=states_geojson,
                height=560,
            )
            render_map_component(folium_hcho, fallback_title="HCHO Hotspot Map", height=560)
        else:
            st.info("No cluster summary data found. Run the HCHO pipeline first.")

    with col_details:
        st.markdown(
            f"""
            <div style="font-size: 14px; font-weight: 700; color: {TEXT_WHITE}; margin-bottom: 6px;">Detected Hotspot Breakdown</div>
            """,
            unsafe_allow_html=True,
        )
        if df_clusters is not None and not df_clusters.empty:
            for _, r in df_clusters.iterrows():
                cid = int(r["cluster_id"])
                lbl = r.get("region_label", f"Cluster {cid}")
                with st.expander(f"📍 {lbl}", expanded=(cid == 0)):
                    st.markdown(
                        f"""
                        <div style="font-size: 13px; line-height: 1.6; color: {TEXT_OFFWHITE};">
                            <b>Centroid:</b> <code>{r['centroid_lat']:.4f}°N, {r['centroid_lon']:.4f}°E</code><br>
                            <b>Spatial Footprint:</b> <b style="color: {ACCENT_SAFFRON};">{int(r['pixel_count'])} pixels (~{int(r['pixel_count'])*100} km²)</b><br>
                            <b>Mean HCHO:</b> <code>{r['mean_hcho_molecules_cm2']:.3e} molec/cm²</code><br>
                            <b>Peak HCHO:</b> <code>{r['max_hcho_molecules_cm2']:.3e} molec/cm²</code><br>
                            <b>Molar Density:</b> <code>{r['mean_hcho_mol_m2']:.4e} mol/m²</code>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No clusters available.")

        # DBSCAN Parameter Summary Card
        st.markdown(
            meta_card(
                title="Spatial Clustering Configuration",
                content=(
                    "<b>Algorithm:</b> Scikit-Learn DBSCAN<br>"
                    "<b>Distance Metric:</b> Haversine spherical distance<br>"
                    f"<b>Neighborhood Radius (ε):</b> {params.get('eps_km', 50.0):.0f} km ({params.get('eps_radians', 0.00785):.5f} rad)<br>"
                    f"<b>Density Cutoff:</b> ≥ {params.get('threshold_percentile', 90.0):.0f}th percentile ({thresh_val:.2e} molec/cm²)<br>"
                    f"<b>Min Core Samples:</b> {params.get('min_samples', 5)} points"
                ),
                border_left_color=ACCENT_SAFFRON,
            ),
            unsafe_allow_html=True,
        )

    # --- DISTRIBUTION & SCIENTIFIC CAVEATS ROW ---
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    c_hist, c_caveat = st.columns([1, 1])

    with c_hist:
        st.markdown(
            f"""
            <div style="font-size: 14px; font-weight: 700; color: {TEXT_WHITE}; margin-bottom: 6px;">HCHO Concentration Distribution & Threshold</div>
            """,
            unsafe_allow_html=True,
        )
        if df_raw_pts is not None and not df_raw_pts.empty:
            thresh = params.get("threshold_value_molecules_cm2", 1.70e16)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_raw_pts["hcho_molecules_cm2"],
                nbinsx=40,
                marker_color=ACCENT_SAFFRON,
                opacity=0.85,
                name="TROPOMI Pixels",
            ))
            fig_hist.add_vline(
                x=thresh,
                line_dash="dash",
                line_color=TEXT_WHITE,
                line_width=1.5,
                annotation_text=f"P90: {thresh:.2e}",
                annotation_position="top right",
                annotation_font_color=TEXT_WHITE,
            )
            apply_plotly_theme(
                fig_hist,
                height=280,
                xaxis_title="HCHO Column Density (molecules/cm²)",
                yaxis_title="Pixel Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Distribution data unavailable.")

    with c_caveat:
        st.markdown(
            f"""
            <div style="font-size: 14px; font-weight: 700; color: {TEXT_WHITE}; margin-bottom: 6px;">Scientific Interpretation & Caveats</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            meta_card(
                title="Critical Photochemical Context",
                content=(
                    "• <b>Concentration vs. Emission:</b> Detected spatial clusters represent elevated atmospheric column density, "
                    "<b>not confirmed point emission stacks</b>. Formaldehyde (HCHO) is an oxidation product of volatile organic compounds.<br>"
                    "• <b>Atmospheric Lifetime:</b> HCHO has an atmospheric lifetime of several hours during daylight (photolysis and OH reaction), "
                    "meaning hotspots reflect regional photochemical production within ~50–100 km.<br>"
                    "• <b>Seasonal Biogenic Emissions:</b> Elevated temperatures increase isoprene emissions from vegetation (particularly central and eastern forests), "
                    "driving biogenic HCHO formation independent of industrial emissions.<br>"
                    "• <b>Cloud Fraction Masking:</b> Pixels with cloud fraction ≥ 0.30 are masked to prevent spectroscopic cloud interference."
                ),
                border_left_color=STATUS_WARNING,
            ),
            unsafe_allow_html=True,
        )
