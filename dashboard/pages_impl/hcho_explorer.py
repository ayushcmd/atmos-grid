"""
HCHO Hotspot Explorer page for AtmosGrid Dashboard.

Displays:
- Interactive geospatial map centered on India with distinct cluster markers.
- Cluster details, coordinates, pixel count, and mean/max concentration.
- Cluster comparison analytics and volume distribution.
- Cluster catalog table with CSV & GeoJSON export download buttons.
- Scientific disclaimer regarding secondary photochemical nature of HCHO.
"""

from __future__ import annotations

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from dashboard.data_loader import (
    PATH_HCHO_GEOJSON,
    load_hcho_benchmark,
    load_hcho_clusters_summary,
    load_hcho_hotspots_geojson,
    load_hcho_report,
)


def render_hcho_explorer():
    st.markdown("### Sentinel-5P / TROPOMI Formaldehyde (HCHO) Hotspot Explorer")
    st.markdown(
        "<p style='color: #8D99AE; margin-top: -10px;'>"
        "Spatial density-based clustering (Haversine DBSCAN) applied to tropospheric HCHO column number density."
        "</p>",
        unsafe_allow_html=True,
    )

    # Scientific Caution Box
    st.markdown(
        """
        <div style="background: rgba(44, 24, 16, 0.6); border-left: 4px solid #F72585; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px; font-size: 13px; color: #F1FAEE;">
            <b>Scientific Caveat & Interpretation:</b> Detected clusters represent spatial concentration patterns of elevated 
            tropospheric formaldehyde column density, <b>not confirmed direct emission sources or fire locations</b>. 
            Formaldehyde is predominantly a secondary photochemical oxidation intermediate formed from volatile organic 
            compounds (biogenic isoprene and anthropogenic VOCs). Chemical lifetime, boundary layer dynamics, 
            and wind advection influence observed vertical column concentrations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_clusters = load_hcho_clusters_summary()
    df_raw_pts = load_hcho_benchmark()
    geojson_data = load_hcho_hotspots_geojson()
    report = load_hcho_report() or {}

    params = report.get("clustering_parameters", {})
    obs_period = report.get("observation_period", "2024-05-01 to 2024-05-31")

    # --- TOP KPI METRIC BAR ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Observation Period</div>
                <div class="kpi-value" style="font-size: 18px; margin-top: 5px;">May 2024</div>
                <div class="kpi-sub">Pre-monsoon photochemical peak</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">DBSCAN Threshold</div>
                <div class="kpi-value" style="font-size: 20px;">P{params.get('threshold_percentile', 90.0):.0f}</div>
                <div class="kpi-sub">≥ {params.get('threshold_value_molecules_cm2', 1.70e16):.2e} molec/cm²</div>
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

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # --- MAP + DETAILS SECTION ---
    col_map, col_details = st.columns([1.3, 0.7])

    with col_map:
        st.markdown("#### Spatial Hotspot Cluster Map")

        # Map layers
        layers = []

        # 1. Background raw observation points
        if df_raw_pts is not None and not df_raw_pts.empty:
            df_bg = df_raw_pts.copy()
            # Mark points as elevated if above threshold
            thresh = params.get("threshold_value_molecules_cm2", 1.70e16)
            df_bg["is_elevated"] = df_bg["hcho_molecules_cm2"] >= thresh

            bg_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_bg,
                get_position="[longitude, latitude]",
                get_color="[72, 229, 194, 60]",
                get_radius=20000,
                pickable=False,
            )
            layers.append(bg_layer)

        # 2. Hotspot Cluster Centroids
        if df_clusters is not None and not df_clusters.empty:
            cluster_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_clusters,
                get_position="[centroid_lon, centroid_lat]",
                get_color="[247, 37, 133, 220]",
                get_radius="pixel_count * 5000 + 40000",
                pickable=True,
                auto_highlight=True,
            )
            layers.append(cluster_layer)

            view_state = pdk.ViewState(
                latitude=23.5,
                longitude=79.0,
                zoom=4.2,
                pitch=25,
            )

            deck = pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>{region_label}</b><br>"
                            "Lat: {centroid_lat:.4f}, Lon: {centroid_lon:.4f}<br>"
                            "<b>Pixel Count:</b> {pixel_count}<br>"
                            "<b>Mean HCHO:</b> {mean_hcho_molecules_cm2:.2e} molec/cm²<br>"
                            "<b>Max HCHO:</b> {max_hcho_molecules_cm2:.2e} molec/cm²"
                },
                map_style="mapbox://styles/mapbox/dark-v10",
            )
            st.pydeck_chart(deck)
        else:
            st.info("No cluster summary data found.")

    with col_details:
        st.markdown("#### Detected Cluster Breakdown")
        if df_clusters is not None and not df_clusters.empty:
            for _, r in df_clusters.iterrows():
                cid = int(r["cluster_id"])
                with st.expander(f"📍 {r['region_label']}", expanded=(cid == 0)):
                    st.markdown(
                        f"""
                        - **Centroid:** `{r['centroid_lat']:.4f}°N, {r['centroid_lon']:.4f}°E`
                        - **Pixel Count:** `{int(r['pixel_count'])} pixels`
                        - **Mean HCHO:** `{r['mean_hcho_molecules_cm2']:.3e} molec/cm²`
                        - **Max HCHO:** `{r['max_hcho_molecules_cm2']:.3e} molec/cm²`
                        - **Molar Density:** `{r['mean_hcho_mol_m2']:.4e} mol/m²`
                        """
                    )
        else:
            st.write("No clusters available.")

    # --- CLUSTER COMPARISON CHARTS ---
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Cluster Analytical Comparison")

    if df_clusters is not None and not df_clusters.empty:
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_bar1 = px.bar(
                df_clusters,
                x="region_label",
                y="mean_hcho_molecules_cm2",
                color="region_label",
                template="plotly_dark",
                labels={"region_label": "Cluster", "mean_hcho_molecules_cm2": "Mean Column Density (molec/cm²)"},
                title="Mean Atmospheric HCHO Concentration",
                color_discrete_sequence=["#F72585", "#7209B7", "#4CC9F0"],
            )
            fig_bar1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                height=320,
                showlegend=False,
                margin=dict(l=30, r=20, t=40, b=50),
            )
            st.plotly_chart(fig_bar1, use_container_width=True)

        with c_chart2:
            fig_bar2 = px.bar(
                df_clusters,
                x="region_label",
                y="pixel_count",
                color="region_label",
                template="plotly_dark",
                labels={"region_label": "Cluster", "pixel_count": "Pixel Footprint Count"},
                title="Spatial Extent / Core Density (Pixel Count)",
                color_discrete_sequence=["#F72585", "#7209B7", "#4CC9F0"],
            )
            fig_bar2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                height=320,
                showlegend=False,
                margin=dict(l=30, r=20, t=40, b=50),
            )
            st.plotly_chart(fig_bar2, use_container_width=True)

    # --- EXPORT & DOWNLOAD SECTION ---
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Export Detected Hotspots")

    col_t, col_d1, col_d2 = st.columns([1.5, 0.5, 0.5])
    with col_t:
        if df_clusters is not None:
            st.dataframe(
                df_clusters[[
                    "cluster_id", "region_label", "centroid_lat", "centroid_lon",
                    "pixel_count", "mean_hcho_molecules_cm2", "max_hcho_molecules_cm2"
                ]],
                use_container_width=True,
                hide_index=True,
            )

    with col_d1:
        if df_clusters is not None:
            csv_bytes = df_clusters.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Summary",
                data=csv_bytes,
                file_name="hcho_clusters_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col_d2:
        if geojson_data is not None:
            geojson_str = json.dumps(geojson_data, indent=2).encode("utf-8")
            st.download_button(
                label="🗺️ Download GeoJSON",
                data=geojson_str,
                file_name="hcho_hotspots.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )
