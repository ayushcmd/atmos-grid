"""
AtmosGrid — Satellite Intelligence for Air Quality and Atmospheric Hotspots.

Main Streamlit Application Entry Point.
"""

from __future__ import annotations

import streamlit as st

# Configure wide layout and page metadata
st.set_page_config(
    page_title="AtmosGrid — Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme CSS (Dark Navy & Slate with Cyan/Teal Accents)
CUSTOM_CSS = """
<style>
    /* Dark Navy Main Background */
    .stApp {
        background-color: #0B132B;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* KPI Cards */
    .kpi-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(72, 229, 194, 0.4);
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 12px;
        color: #64748B;
        margin-top: 6px;
    }

    /* Information / Metadata Card */
    .meta-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 6px;
        padding: 14px 16px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: #94A3B8;
        font-weight: 500;
        font-size: 14px;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(72, 229, 194, 0.08) !important;
        color: #48E5C2 !important;
        border-bottom: 2px solid #48E5C2 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1E293B !important;
        border-radius: 6px;
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
    # --- SIDEBAR BRANDING & NAVIGATION ---
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 0 20px 0;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: linear-gradient(135deg, #48E5C2 0%, #00B4D8 100%); width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px;">
                        🌐
                    </div>
                    <div>
                        <span style="font-size: 22px; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF;">AtmosGrid</span>
                        <div style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #48E5C2; font-weight: 600;">Phase 1.5 Prototype</div>
                    </div>
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-top: 8px; line-height: 1.35;">
                    Satellite Intelligence for Air Quality & Atmospheric Hotspots over India.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("<div style='font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 8px;'>Navigation</div>", unsafe_allow_html=True)

        page_selection = st.radio(
            label="Dashboard Pages",
            options=[
                "Overview",
                "AQI Analytics",
                "HCHO Hotspot Explorer",
                "Data Quality & Methodology",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 11px; color: #64748B; line-height: 1.5;">
                <b>System Version:</b> v1.5.0<br>
                <b>ML Engine:</b> XGBoost Regressor (3.4.1)<br>
                <b>Spatial Model:</b> Haversine DBSCAN<br>
                <b>Ground Data:</b> OpenAQ S3 / CPCB<br>
                <b>Satellite:</b> Sentinel-5P TROPOMI
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- MAIN ROUTER ---
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

    # Footer
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
