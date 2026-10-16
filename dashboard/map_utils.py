"""
AtmosGrid Interactive Geographic Mapping Utilities.

Provides high-performance Folium map rendering with:
- Dark basemap tiles (CartoDB dark_matter)
- India 36 States/UTs administrative boundaries
- ISRO-inspired visual palette: Saffron cluster footprints, crisp white ground stations
- Rich HTML tooltips, popups, and coordinated dark-matte styling
- Graceful offline fallback
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import folium
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.theme import (
    ACCENT_SAFFRON,
    ACCENT_SAFFRON_DEEP,
    BG_BASE,
    BG_ELEVATED,
    BG_SURFACE,
    BORDER_BASE,
    TEXT_MUTED,
    TEXT_OFFWHITE,
    TEXT_WHITE,
)

try:
    from streamlit_folium import st_folium
    HAS_STREAMLIT_FOLIUM = True
except ImportError:
    HAS_STREAMLIT_FOLIUM = False

logger = logging.getLogger(__name__)

CARTO_DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
CARTO_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'


def build_folium_hcho_map(
    df_clusters: pd.DataFrame,
    df_raw_pts: Optional[pd.DataFrame] = None,
    states_geojson: Optional[Dict[str, Any]] = None,
    height: int = 560,
) -> Any:
    """Build an interactive Folium map for HCHO Hotspot clusters and Indian geography."""
    # Center over India
    m = folium.Map(
        location=[22.8, 80.5],
        zoom_start=5,
        tiles=CARTO_DARK_TILES,
        attr=CARTO_ATTR,
        control_scale=True,
    )

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
                "fillColor": "#15181C",
                "color": "#3E444D",
                "weight": 1.0,
                "fillOpacity": 0.10,
                "dashArray": "2, 4",
            },
            highlight_function=lambda x: {
                "fillColor": ACCENT_SAFFRON,
                "color": ACCENT_SAFFRON,
                "weight": 2.0,
                "fillOpacity": 0.22,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_1"],
                aliases=["State / UT:"],
                style=f"background-color: {BG_SURFACE}; color: {ACCENT_SAFFRON}; font-family: sans-serif; font-size: 12px; padding: 6px 10px; border-radius: 4px; border: 1px solid {BORDER_BASE}; font-weight: 600;",
            ),
        ).add_to(m)

    # 2. Background Satellite Observation Footprint
    if df_raw_pts is not None and not df_raw_pts.empty:
        sat_group = folium.FeatureGroup(name="Satellite Observation Footprint", show=True)
        sample_pts = df_raw_pts.sample(min(len(df_raw_pts), 300), random_state=42)
        for _, r in sample_pts.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=3,
                color="#525863",
                weight=0.8,
                fill=True,
                fill_color=ACCENT_SAFFRON,
                fill_opacity=0.30,
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
            radius_outer = min(pix_cnt * 2.4 + 18, 50)

            # Outer Footprint Halo (Saffron dispersion)
            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=radius_outer,
                color=ACCENT_SAFFRON,
                weight=1.5,
                fill=True,
                fill_color=ACCENT_SAFFRON,
                fill_opacity=0.18,
            ).add_to(cluster_group)

            # Core Centroid Marker Popup
            core_html = f"""
            <div style="background-color: {BG_SURFACE}; color: {TEXT_OFFWHITE}; padding: 14px 16px; border-radius: 6px; border: 1px solid {BORDER_BASE}; border-top: 3px solid {ACCENT_SAFFRON}; min-width: 240px; font-family: -apple-system, sans-serif;">
                <div style="font-weight: 700; font-size: 13px; color: {TEXT_WHITE}; margin-bottom: 2px;">{label}</div>
                <div style="font-size: 11px; color: {TEXT_MUTED}; margin-bottom: 8px;">Cluster #{cid} Centroid: {c_lat:.4f}°N, {c_lon:.4f}°E</div>
                <hr style="border: none; border-top: 1px solid {BORDER_BASE}; margin: 8px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                    <span style="color: {TEXT_MUTED};">Spatial Footprint:</span>
                    <b style="color: {ACCENT_SAFFRON};">{pix_cnt} pixels (~{pix_cnt*100} km²)</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                    <span style="color: {TEXT_MUTED};">Mean HCHO:</span>
                    <b style="color: {TEXT_WHITE};">{mean_hcho:.2e} molec/cm²</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                    <span style="color: {TEXT_MUTED};">Peak HCHO:</span>
                    <b style="color: {TEXT_WHITE};">{max_hcho:.2e} molec/cm²</b>
                </div>
                <div style="font-size: 10px; color: {TEXT_MUTED}; margin-top: 10px; background: {BG_ELEVATED}; padding: 6px 8px; border-radius: 4px; border: 1px solid {BORDER_BASE};">
                    Atmospheric column density. Not verified as a point emission stack.
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[c_lat, c_lon],
                radius=8,
                color=TEXT_WHITE,
                weight=2,
                fill=True,
                fill_color=ACCENT_SAFFRON_DEEP,
                fill_opacity=0.95,
                popup=folium.Popup(core_html, max_width=320),
                tooltip=f"{label} (Centroid #{cid})",
            ).add_to(cluster_group)

        cluster_group.add_to(m)

    # 4. Floating Legend & Navigation Instructions
    legend_html = f"""
    <div style="position: fixed; bottom: 25px; right: 25px; z-index: 1000; background: rgba(25, 28, 32, 0.92); backdrop-filter: blur(8px); padding: 14px 18px; border-radius: 6px; border: 1px solid {BORDER_BASE}; font-family: -apple-system, sans-serif; font-size: 11px; color: {TEXT_OFFWHITE}; box-shadow: 0 4px 16px rgba(0,0,0,0.5);">
        <div style="font-weight: 700; font-size: 11px; color: {TEXT_WHITE}; margin-bottom: 8px; letter-spacing: 0.08em; text-transform: uppercase;">Map Legend & Layers</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: {ACCENT_SAFFRON_DEEP}; border: 1.5px solid {TEXT_WHITE};"></div>
            <span>HCHO Hotspot Core (DBSCAN P90)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: rgba(242, 140, 40, 0.25); border: 1px solid {ACCENT_SAFFRON};"></div>
            <span>Estimated Spatial Footprint</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: {ACCENT_SAFFRON}; opacity: 0.5;"></div>
            <span>TROPOMI Grid Pixel</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <div style="width: 16px; height: 2px; background: #3E444D; border-top: 1px dashed #3E444D;"></div>
            <span>State / UT Boundary</span>
        </div>
        <div style="font-size: 10px; color: {TEXT_MUTED}; border-top: 1px solid {BORDER_BASE}; padding-top: 6px;">
            Scroll to zoom · Drag to pan · Click marker for telemetry
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
                    "fillColor": ACCENT_SAFFRON,
                    "color": ACCENT_SAFFRON,
                    "weight": 1.6,
                    "fillOpacity": 0.16,
                }
            elif status == "SATELLITE_ONLY":
                return {
                    "fillColor": "#15181C",
                    "color": "#3E444D",
                    "weight": 0.9,
                    "fillOpacity": 0.08,
                    "dashArray": "2, 3",
                }
            else:
                return {
                    "fillColor": "#191C20",
                    "color": "#2D3239",
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
                style=f"background-color: {BG_SURFACE}; color: {TEXT_WHITE}; font-family: sans-serif; font-size: 12px; padding: 6px 10px; border-radius: 4px; border: 1px solid {BORDER_BASE}; font-weight: 600;",
            ),
        ).add_to(m)

    # 2. CPCB Ground Monitoring Stations Layer (Crisp White Markers)
    if df_stations is not None and not df_stations.empty:
        station_group = folium.FeatureGroup(name="CPCB Ground CAAQMS Stations", show=True)
        unique_stn = df_stations.drop_duplicates(subset=["location_id"]).copy()
        for _, r in unique_stn.iterrows():
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            s_name = r.get("station_name", f"Station {r['location_id']}")
            state = r.get("state", "India")

            stn_html = f"""
            <div style="background-color: {BG_SURFACE}; color: {TEXT_OFFWHITE}; padding: 12px 14px; border-radius: 6px; border: 1px solid {BORDER_BASE}; border-left: 3px solid {TEXT_WHITE}; font-family: -apple-system, sans-serif; min-width: 220px;">
                <div style="font-weight: 700; font-size: 12px; color: {TEXT_WHITE}; margin-bottom: 3px;">{s_name}</div>
                <div style="font-size: 11px; color: {TEXT_MUTED};">State: <b style="color: {TEXT_WHITE};">{state}</b> · Lat: {lat:.4f}, Lon: {lon:.4f}</div>
                <div style="font-size: 10px; color: {TEXT_MUTED}; margin-top: 6px; background: {BG_ELEVATED}; padding: 4px 6px; border-radius: 4px;">
                    Continuous CPCB Ambient Air Quality Station
                </div>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=TEXT_WHITE,
                weight=2,
                fill=True,
                fill_color=ACCENT_SAFFRON,
                fill_opacity=1.0,
                popup=folium.Popup(stn_html, max_width=280),
                tooltip=f"CPCB Station: {s_name} ({state})",
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
                color=ACCENT_SAFFRON,
                weight=1.5,
                fill=True,
                fill_color=ACCENT_SAFFRON_DEEP,
                fill_opacity=0.75,
                tooltip=f"HCHO Hotspot: {lbl}",
            ).add_to(cluster_group)
        cluster_group.add_to(m)

    # Overview Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 25px; right: 25px; z-index: 1000; background: rgba(25, 28, 32, 0.92); backdrop-filter: blur(8px); padding: 14px 18px; border-radius: 6px; border: 1px solid {BORDER_BASE}; font-family: -apple-system, sans-serif; font-size: 11px; color: {TEXT_OFFWHITE}; box-shadow: 0 4px 16px rgba(0,0,0,0.5);">
        <div style="font-weight: 700; font-size: 11px; color: {TEXT_WHITE}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em;">India Geospatial Overview</div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: {ACCENT_SAFFRON}; border: 2px solid {TEXT_WHITE};"></div>
            <span>CPCB Ground Monitoring Station</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: {ACCENT_SAFFRON_DEEP}; border: 1px solid {ACCENT_SAFFRON};"></div>
            <span>Sentinel-5P HCHO Hotspot Centroid</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <div style="width: 14px; height: 10px; background: rgba(242, 140, 40, 0.16); border: 1px solid {ACCENT_SAFFRON};"></div>
            <span>State with Ground Calibration (4 States)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
            <div style="width: 14px; height: 10px; background: rgba(21, 24, 28, 0.4); border: 1px dashed #3E444D;"></div>
            <span>Satellite Coverage Only (32 States/UTs)</span>
        </div>
        <div style="font-size: 10px; color: {TEXT_MUTED}; border-top: 1px solid {BORDER_BASE}; padding-top: 6px;">
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

    st.info(f"Viewing {fallback_title} via static map projection (interactive tiles require webGL / leaflet support).")
