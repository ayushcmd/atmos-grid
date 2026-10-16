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

import json
import pandas as pd
import plotly.express as px
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


def render_hcho_explorer():
    st.markdown("### Sentinel-5P HCHO Hotspot Explorer")
    st.markdown(
        "<p style='color: #94A3B8; margin-top: -10px; font-size: 14px;'>"
        "Spatial density-based clustering (Haversine DBSCAN) applied to TROPOMI formaldehyde tropospheric column measurements over India."
        "</p>",
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
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Identified Hotspots</div>
                <div class="kpi-value">{report.get('num_clusters', 3)} <span style="font-size: 14px; color: #F72585;">clusters</span></div>
                <div class="kpi-sub">Haversine DBSCAN Spatial Clustering</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        thresh_val = params.get("threshold_value_molecules_cm2", 1.70e16)
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">DBSCAN Threshold</div>
                <div class="kpi-value" style="font-size: 24px;">P{params.get('threshold_percentile', 90.0):.0f}</div>
                <div class="kpi-sub">≥ {thresh_val:.2e} molec/cm²</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Neighborhood Radius</div>
                <div class="kpi-value">{params.get('eps_km', 50.0):.0f} <span style="font-size: 14px; color: #48E5C2;">km</span></div>
                <div class="kpi-sub">Haversine metric (min_samples=5)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Clustered Footprint</div>
                <div class="kpi-value">{report.get('clustered_pixels', 29)} <span style="font-size: 14px; color: #48E5C2;">pixels</span></div>
                <div class="kpi-sub">Across {report.get('num_clusters', 3)} distinct spatial cores</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # --- MAP + DETAILS SECTION ---
    col_map, col_details = st.columns([1.25, 0.75])

    with col_map:
        st.markdown("##### Spatial Hotspot Cluster Map")
        st.caption("Interactive basemap powered by CartoDB Dark Matter with Indian administrative boundaries and HCHO cluster footprints.")

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
        st.markdown("##### Detected Cluster Breakdown")
        if df_clusters is not None and not df_clusters.empty:
            for _, r in df_clusters.iterrows():
                cid = int(r["cluster_id"])
                lbl = r.get("region_label", f"Cluster {cid}")
                with st.expander(f"📍 {lbl}", expanded=(cid == 0)):
                    st.markdown(
                        f"""
                        <div style="font-size: 13px; line-height: 1.6; color: #E2E8F0;">
                            <b>Centroid:</b> <code>{r['centroid_lat']:.4f}°N, {r['centroid_lon']:.4f}°E</code><br>
                            <b>Spatial Footprint:</b> <b style="color: #48E5C2;">{int(r['pixel_count'])} pixels (~{int(r['pixel_count'])*100} km²)</b><br>
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
            f"""
            <div class="meta-card" style="margin-top: 15px;">
                <div style="font-weight: 600; color: #48E5C2; margin-bottom: 6px;">Spatial Clustering Architecture</div>
                <div style="font-size: 12px; color: #94A3B8; line-height: 1.5;">
                    <b>Algorithm:</b> Scikit-Learn DBSCAN<br>
                    <b>Distance Metric:</b> Haversine spherical distance<br>
                    <b>Neighborhood Radius (ε):</b> {params.get('eps_km', 50.0):.0f} km ({params.get('eps_radians', 0.00785):.5f} rad)<br>
                    <b>Density Cutoff:</b> ≥ {params.get('threshold_percentile', 90.0):.0f}th percentile ({thresh_val:.2e} molec/cm²)<br>
                    <b>Min Core Samples:</b> {params.get('min_samples', 5)} points
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- DISTRIBUTION & SCIENTIFIC CAVEATS ROW ---
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    c_hist, c_caveat = st.columns([1, 1])

    with c_hist:
        st.markdown("##### HCHO Concentration Distribution & Threshold")
        if df_raw_pts is not None and not df_raw_pts.empty:
            thresh = params.get("threshold_value_molecules_cm2", 1.70e16)
            fig_hist = px.histogram(
                df_raw_pts,
                x="hcho_molecules_cm2",
                nbins=40,
                color_discrete_sequence=["#00B4D8"],
                template="plotly_dark",
                labels={"hcho_molecules_cm2": "HCHO Column Density (molecules/cm²)"},
            )
            # Add vertical threshold line
            fig_hist.add_vline(
                x=thresh,
                line_dash="dash",
                line_color="#F72585",
                line_width=2,
                annotation_text=f"P90 Threshold: {thresh:.2e}",
                annotation_position="top right",
                annotation_font_color="#F72585",
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                margin=dict(l=30, r=20, t=30, b=30),
                height=280,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="Pixel Count"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Distribution data unavailable.")

    with c_caveat:
        st.markdown("##### Scientific Interpretation & Caveats")
        st.markdown(
            """
            <div class="meta-card" style="height: 280px; overflow-y: auto;">
                <b style="color: #F72585;">Important Scientific Caveats:</b>
                <ul style="font-size: 12px; color: #94A3B8; line-height: 1.6; margin-top: 8px; padding-left: 18px;">
                    <li><b>Concentration vs. Emission:</b> Detected spatial clusters represent elevated atmospheric column density, <b>not confirmed point emission stacks</b>. Formaldehyde (HCHO) is a secondary intermediate of VOC oxidation.</li>
                    <li><b>Atmospheric Lifetime:</b> HCHO has an atmospheric lifetime of several hours during daylight (photolysis and OH reaction), meaning hotspots reflect regional photochemical production and biogenic/anthropogenic precursors within ~50–100 km.</li>
                    <li><b>Seasonal Biogenic Emissions:</b> High temperatures elevate isoprene emissions from vegetation (especially central and eastern forests), driving biogenic HCHO formation independent of industrial emissions.</li>
                    <li><b>Cloud Fraction Masking:</b> Pixels with cloud fraction ≥ 0.30 are masked to prevent spectroscopic cloud interference.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
