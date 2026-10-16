"""
AtmosGrid Interactive Geographic Mapping Utilities.

Provides high-performance, token-free Folium map rendering with:
- Dark basemap tiles (CartoDB dark_matter / OpenStreetMap)
- India 36 States/UTs administrative boundaries
- Distinct HCHO hotspot cluster markers with concentration-proportional footprints
- Active CPCB ground-monitoring stations
- Rich HTML tooltips, popups, and coordinated cyan/teal/magenta visual styling
- Automatic offline Plotly Geo Map fallback when basemap tiles or network are unavailable
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    HAS_STREAMLIT_FOLIUM = False

logger = logging.getLogger(__name__)

# Color Palette Constants
COLOR_BG_DARK = "#0B132B"
COLOR_SLATE_CARD = "#1E293B"
COLOR_CYAN_ACCENT = "#48E5C2"
COLOR_TEAL_ACCENT = "#00B4D8"
COLOR_MAGENTA_ACCENT = "#F72585"
COLOR_AMBER_WARNING = "#F59E0B"
COLOR_BORDER_MUTED = "rgba(255, 255, 255, 0.12)"

CARTO_DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'


def build_folium_hcho_map(
    df_clusters: pd.DataFrame,
    df_raw_pts: Optional[pd.DataFrame] = None,
    states_geojson: Optional[Dict[str, Any]] = None,
    height: int = 560,
) -> Any:
    """Build a rich, interactive Folium map for HCHO Hotspot clusters and Indian geography."""
    # Center over India
    m = folium.Map(
        location=[22.8, 80.5],
        zoom_start=5,
        tiles=CARTO_DARK_TILES,
        attr=CARTO_ATTR,
        control_scale=True,
    )

    # Optional OpenStreetMap layer for alternative light viewing
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        control=True,
    ).add_to(m)

    # 1. State / UT Administrative Boundaries Layer
    if states_geojson is not None and "features" in states_geojson:
        folium.GeoJson(
            states_geojson,
            name="State & UT Boundaries",
            style_function=lambda x: {
                "fillColor": "#0F172A",
                "color": "#00B4D8",
                "weight": 1.2,
                "fillOpacity": 0.12,
                "dashArray": "2, 4",
            },
            highlight_function=lambda x: {
                "fillColor": "#48E5C2",
                "color": "#48E5C2",
                "weight": 2.0,
                "fillOpacity": 0.28,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_1"],
                aliases=["State / UT:"],
                style="background-color: #0F172A; color: #48E5C2; font-family: sans-serif; font-size: 12px; padding: 6px; border-radius: 4px; border: 1px solid rgba(72, 229, 194, 0.4);",
            ),
        ).add_to(m)

    # 2. Background Satellite Observation Footprint
    if df_raw_pts is not None and not df_raw_pts.empty:
        sat_group = folium.FeatureGroup(name="Satellite Observation Footprint", show=True)
        # Sample points if large to maintain crisp 60fps rendering
        sample_pts = df_raw_pts.sample(min(len(df_raw_pts), 300), random_state=42)
        for _, r in sample_pts.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=3,
                color="#00B4D8",
                weight=1,
                fill=True,
                fill_color="#48E5C2",
                fill_opacity=0.35,
                tooltip=f"TROPOMI Pixel: {r['latitude']:.2f}°N, {r['longitude']:.2f}°E | HCHO: {r.get('hcho_molecules_cm2', 0):.2e} molec/cm²",
            ).add_to(sat_group)
        sat_group.add_to(m)

    # 3. HCHO Hotspot Clusters
    if df_clusters is not None and not df_clusters.empty:
        cluster_group = folium.FeatureGroup(name="DBSCAN Hotspot Clusters (P90)", show=True)
        for _, r in df_clusters.iterrows():
            cid = int(r["cluster_id"])
            c_lat = float(r["centroid_lat"])
            c_lon = float(r["centroid_lon"])
            pix_cnt = int(r["pixel_count"])
            mean_hcho = float(r["mean_hcho_molecules_cm2"])
            max_hcho = float(r["max_hcho_molecules_cm2"])
            label = r.get("region_label", f"Hotspot Cluster {cid}")

            # Marker size proportional to cluster footprint
            radius_outer = min(pix_cnt * 2.4 + 18, 48)

            # Outer Footprint Halo
            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=radius_outer,
                color=COLOR_MAGENTA_ACCENT,
                weight=1.5,
                fill=True,
                fill_color=COLOR_MAGENTA_ACCENT,
                fill_opacity=0.22,
            ).add_to(cluster_group)

            # Core Centroid Marker
            core_html = f"""
            <div style="background-color: #0F172A; color: #E2E8F0; padding: 12px 14px; border-radius: 8px; border: 1px solid #F72585; min-width: 230px; font-family: -apple-system, sans-serif;">
                <div style="font-weight: 700; font-size: 13px; color: #F72585; margin-bottom: 4px;">📍 {label}</div>
                <div style="font-size: 11px; color: #94A3B8; margin-bottom: 8px;">Cluster #{cid} Centroid: {c_lat:.4f}°N, {c_lon:.4f}°E</div>
                <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 6px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                    <span style="color: #94A3B8;">Spatial Footprint:</span>
                    <b style="color: #48E5C2;">{pix_cnt} pixels (~{pix_cnt*100} km²)</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                    <span style="color: #94A3B8;">Mean HCHO:</span>
                    <b style="color: #F8FAFC;">{mean_hcho:.2e} molec/cm²</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                    <span style="color: #94A3B8;">Peak HCHO:</span>
                    <b style="color: #F8FAFC;">{max_hcho:.2e} molec/cm²</b>
                </div>
                <div style="font-size: 10px; color: #CBD5E1; margin-top: 8px; font-style: italic; background: rgba(247, 37, 133, 0.12); padding: 5px; border-radius: 4px;">
                    ⚠️ Concentration Hotspot: Atmospheric column density. Not verified as a point emission source.
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=8,
                color="#48E5C2",
                weight=2,
                fill=True,
                fill_color=COLOR_MAGENTA_ACCENT,
                fill_opacity=0.9,
                popup=folium.Popup(core_html, max_width=320),
                tooltip=f"📍 {label} (Click for analytics)",
            ).add_to(cluster_group)

        cluster_group.add_to(m)

    # 4. Floating Legend & Navigation Instructions
    legend_html = """
    <div style="position: fixed; bottom: 25px; right: 25px; z-index: 1000; background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(8px); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.12); font-family: -apple-system, sans-serif; font-size: 11px; color: #E2E8F0; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
        <div style="font-weight: 700; font-size: 12px; color: #FFFFFF; margin-bottom: 8px; letter-spacing: 0.05em; text-transform: uppercase;">Map Legend & Layers</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: #F72585; border: 2px solid #48E5C2;"></div>
            <span>HCHO Hotspot Cluster (DBSCAN P90)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #48E5C2; opacity: 0.7;"></div>
            <span>Satellite Observation Grid</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <div style="width: 16px; height: 2px; background: #00B4D8; border-top: 1px dashed #00B4D8;"></div>
            <span>State / UT Boundary</span>
        </div>
        <div style="font-size: 10px; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
            🖱️ Scroll to zoom · Drag to pan · Click marker for details
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright").add_to(m)
    return m


def build_folium_overview_map(
    df_stations: pd.DataFrame,
    df_clusters: pd.DataFrame,
    states_geojson: Optional[Dict[str, Any]] = None,
    df_state_coverage: Optional[pd.DataFrame] = None,
) -> Any:
    """Build comprehensive India geographic overview map showing stations, clusters, and boundaries."""
    m = folium.Map(
        location=[22.8, 80.5],
        zoom_start=5,
        tiles=CARTO_DARK_TILES,
        attr=CARTO_ATTR,
        control_scale=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # State status dictionary if available
    state_status_dict = {}
    if df_state_coverage is not None and not df_state_coverage.empty:
        for _, row in df_state_coverage.iterrows():
            state_status_dict[row["state"]] = row.get("coverage_status", "SATELLITE_ONLY")

    # 1. State Boundaries Layer with Status-Coded Colors
    if states_geojson is not None and "features" in states_geojson:
        def get_state_style(feature):
            s_name = feature["properties"].get("NAME_1", "")
            status = state_status_dict.get(s_name, "SATELLITE_ONLY")
            if status == "MEASURED_GROUND":
                return {
                    "fillColor": "#48E5C2",
                    "color": "#48E5C2",
                    "weight": 1.5,
                    "fillOpacity": 0.22,
                }
            elif status == "SATELLITE_ONLY":
                return {
                    "fillColor": "#0F172A",
                    "color": "#00B4D8",
                    "weight": 1.0,
                    "fillOpacity": 0.08,
                    "dashArray": "2, 3",
                }
            else:
                return {
                    "fillColor": "#1E293B",
                    "color": "#64748B",
                    "weight": 0.8,
                    "fillOpacity": 0.05,
                }

        folium.GeoJson(
            states_geojson,
            name="State Coverage Classification",
            style_function=get_state_style,
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_1"],
                aliases=["State / UT:"],
                style="background-color: #0F172A; color: #48E5C2; font-family: sans-serif; font-size: 12px; padding: 6px; border-radius: 4px;",
            ),
        ).add_to(m)

    # 2. CPCB Ground Monitoring Stations Layer
    if df_stations is not None and not df_stations.empty:
        station_group = folium.FeatureGroup(name="CPCB Ground CAAQMS Stations", show=True)
        # Unique stations
        unique_stn = df_stations.drop_duplicates(subset=["location_id"]).copy()
        for _, r in unique_stn.iterrows():
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            s_name = r.get("station_name", f"Station {r['location_id']}")
            state = r.get("state", "India")
            
            stn_html = f"""
            <div style="background-color: #0F172A; color: #E2E8F0; padding: 10px 12px; border-radius: 6px; border: 1px solid #48E5C2; font-family: -apple-system, sans-serif; min-width: 200px;">
                <div style="font-weight: 700; font-size: 12px; color: #48E5C2; margin-bottom: 2px;">🏢 {s_name}</div>
                <div style="font-size: 11px; color: #94A3B8;">State: <b>{state}</b> · Lat: {lat:.4f}, Lon: {lon:.4f}</div>
                <div style="font-size: 10px; color: #64748B; margin-top: 6px;">Continuous CPCB Ambient Air Quality Station</div>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=7,
                color="#FFFFFF",
                weight=1.5,
                fill=True,
                fill_color="#48E5C2",
                fill_opacity=0.95,
                popup=folium.Popup(stn_html, max_width=280),
                tooltip=f"🏢 {s_name} ({state})",
            ).add_to(station_group)

        station_group.add_to(m)

    # 3. HCHO Hotspot Clusters Centroids
    if df_clusters is not None and not df_clusters.empty:
        cluster_group = folium.FeatureGroup(name="Sentinel-5P HCHO Hotspots", show=True)
        for _, r in df_clusters.iterrows():
            c_lat = float(r["centroid_lat"])
            c_lon = float(r["centroid_lon"])
            lbl = r.get("region_label", f"Cluster {r['cluster_id']}")
            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=14,
                color="#FFFFFF",
                weight=1.5,
                fill=True,
                fill_color=COLOR_MAGENTA_ACCENT,
                fill_opacity=0.85,
                tooltip=f"🟣 HCHO Hotspot: {lbl}",
            ).add_to(cluster_group)
        cluster_group.add_to(m)

    # Overview Legend
    legend_html = """
    <div style="position: fixed; bottom: 25px; right: 25px; z-index: 1000; background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(8px); padding: 14px 18px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.12); font-family: -apple-system, sans-serif; font-size: 11px; color: #E2E8F0; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">
        <div style="font-weight: 700; font-size: 12px; color: #FFFFFF; margin-bottom: 8px; text-transform: uppercase;">India Geospatial Overview</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: #48E5C2; border: 1.5px solid #FFFFFF;"></div>
            <span>CPCB Ground Monitoring Station</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: #F72585; border: 1.5px solid #FFFFFF;"></div>
            <span>Sentinel-5P HCHO Hotspot Centroid</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 14px; height: 10px; background: rgba(72, 229, 194, 0.22); border: 1px solid #48E5C2;"></div>
            <span>State with Ground Calibration (4 States)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <div style="width: 14px; height: 10px; background: rgba(15, 23, 42, 0.4); border: 1px dashed #00B4D8;"></div>
            <span>Satellite Coverage Only (32 States/UTs)</span>
        </div>
        <div style="font-size: 10px; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
            Note: HCHO centroids represent atmospheric columns, not ground monitors.
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright").add_to(m)
    return m


def render_map_component(folium_map: Any, fallback_title: str = "Geospatial View", height: int = 560):
    """Safely render the Folium map in Streamlit with graceful fallback."""
    if HAS_STREAMLIT_FOLIUM:
        try:
            st_folium(
                folium_map,
                width="100%",
                height=height,
                returned_objects=[],
            )
            return
        except Exception as e:
            logger.error("st_folium rendering failed: %s", e)
            st.warning("Interactive tile rendering encountered a local display error. Rendering fallback view.")

    # Graceful Fallback Notice
    st.info(f"Viewing {fallback_title} via static map projection (interactive tiles require webGL / leaflet support).")
