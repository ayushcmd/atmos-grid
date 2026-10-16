"""
AtmosGrid UI Theme and Design System.

ISRO-inspired visual tokens: Saffron, Deep Black, and Crisp White.
Provides scoped CSS, layout constants, and Plotly styling helpers
for modern scientific and aerospace-grade Earth-observation dashboards.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import plotly.graph_objects as go

# --- COLOR DESIGN TOKENS ---
BG_BASE = "#111315"           # Near-black canvas
BG_SURFACE = "#191C20"        # Card & container surface
BG_ELEVATED = "#22262B"       # Elevated panels & hover states
BG_HEADER = "#15181C"         # Mission control header

ACCENT_SAFFRON = "#F28C28"      # Primary ISRO-inspired saffron
ACCENT_SAFFRON_DEEP = "#D96B12" # Secondary deep saffron
ACCENT_SAFFRON_GLOW = "rgba(242, 140, 40, 0.20)"

TEXT_WHITE = "#FFFFFF"        # Headings & primary numerals
TEXT_OFFWHITE = "#F7F7F5"     # Primary body text
TEXT_MUTED = "#A7ADB5"        # Secondary labels & subtitles
TEXT_DIM = "#6B7280"          # Tertiary notes & timestamps

BORDER_BASE = "#2D3239"       # Neutral surface border
BORDER_MUTED = "#22262B"      # Subtle divider border
BORDER_ACCENT = "rgba(242, 140, 40, 0.40)"

STATUS_SUCCESS = "#22C55E"    # Validated / Active
STATUS_WARNING = "#F28C28"    # Warning / Elevated
STATUS_INFO = "#38BDF8"       # Information / Satellite only
STATUS_CRITICAL = "#EF4444"   # Severe / Excluded

# Font stack
FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
    'Helvetica, Arial, sans-serif'
)

# --- SCOPED APPLICATION STYLESHEET ---
THEME_CSS = f"""
<style>
    /* 1. Global Application Background & Base Typography */
    .stApp {{
        background-color: {BG_BASE};
        color: {TEXT_OFFWHITE};
        font-family: {FONT_FAMILY};
    }}

    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    /* Hide standard sidebar collapse chevron decoration if sidebar collapsed */
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    /* 2. Mission Control Header Bar */
    .atmos-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 22px;
        background: {BG_HEADER};
        border-radius: 8px;
        border: 1px solid {BORDER_BASE};
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }}

    .atmos-logo-icon {{
        background: linear-gradient(135deg, {ACCENT_SAFFRON} 0%, {ACCENT_SAFFRON_DEEP} 100%);
        width: 38px;
        height: 38px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 2px 8px rgba(242, 140, 40, 0.35);
    }}

    .atmos-badge {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {ACCENT_SAFFRON};
        font-weight: 700;
        background: rgba(242, 140, 40, 0.12);
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(242, 140, 40, 0.3);
    }}

    .atmos-telemetry-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        color: {TEXT_MUTED};
        background: {BG_SURFACE};
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid {BORDER_BASE};
    }}

    /* 3. Horizontal Segmented Mission Navigation Bar */
    .stRadio {{
        margin-bottom: 20px;
    }}
    div[data-testid="stRadio"] > div {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        background: {BG_SURFACE};
        padding: 6px;
        border-radius: 8px;
        border: 1px solid {BORDER_BASE};
    }}
    div[data-testid="stRadio"] label {{
        flex: 1 1 auto !important;
        min-width: 160px;
        text-align: center;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        cursor: pointer !important;
        transition: all 0.18s ease-in-out !important;
        color: {TEXT_MUTED} !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    /* Hide the radio circle icon */
    div[data-testid="stRadio"] label > div:first-child {{
        display: none !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        background: {BG_ELEVATED} !important;
        color: {TEXT_WHITE} !important;
        border-color: rgba(242, 140, 40, 0.3) !important;
    }}
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] input:checked + div {{
        background: {BG_ELEVATED} !important;
        border: 1px solid {ACCENT_SAFFRON} !important;
        color: {TEXT_WHITE} !important;
        box-shadow: 0 0 12px {ACCENT_SAFFRON_GLOW} !important;
    }}

    /* 4. Scientific KPI Card System */
    .kpi-card {{
        background: {BG_SURFACE};
        border: 1px solid {BORDER_BASE};
        border-top: 2px solid {ACCENT_SAFFRON};
        border-radius: 6px;
        padding: 16px 18px;
        box-shadow: 0 4px 8px -2px rgba(0, 0, 0, 0.4);
        transition: transform 0.15s ease, border-color 0.15s ease;
        position: relative;
    }}
    .kpi-card:hover {{
        border-color: {ACCENT_SAFFRON};
        transform: translateY(-2px);
    }}
    .kpi-label {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {TEXT_MUTED};
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 26px;
        font-weight: 800;
        color: {TEXT_WHITE};
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
    }}
    .kpi-sub {{
        font-size: 11px;
        color: {TEXT_MUTED};
        margin-top: 6px;
    }}

    /* 5. Metadata & Diagnostic Card */
    .meta-card {{
        background: {BG_SURFACE};
        border: 1px solid {BORDER_BASE};
        border-radius: 6px;
        padding: 14px 16px;
        color: {TEXT_OFFWHITE};
        font-size: 13px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    }}

    /* 6. Tabs System */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {BORDER_BASE};
        margin-bottom: 16px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: {TEXT_MUTED};
        font-weight: 600;
        font-size: 13px;
        padding: 0 16px;
        transition: color 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {TEXT_WHITE};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(242, 140, 40, 0.08) !important;
        color: {TEXT_WHITE} !important;
        border-bottom: 2px solid {ACCENT_SAFFRON} !important;
    }}

    /* 7. Form Controls & Selectboxes */
    div[data-baseweb="select"] > div {{
        background-color: {BG_SURFACE} !important;
        border-color: {BORDER_BASE} !important;
        color: {TEXT_OFFWHITE} !important;
        border-radius: 6px !important;
    }}
    div[data-baseweb="select"]:hover > div {{
        border-color: {ACCENT_SAFFRON} !important;
    }}

    /* 8. Expanders */
    div[data-testid="stExpander"] {{
        background-color: {BG_SURFACE} !important;
        border: 1px solid {BORDER_BASE} !important;
        border-radius: 6px !important;
        margin-bottom: 8px !important;
    }}
    div[data-testid="stExpander"] summary {{
        color: {TEXT_WHITE} !important;
        font-weight: 600 !important;
    }}

    /* 9. DataFrames & Tables */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER_BASE};
        border-radius: 6px;
        overflow: hidden;
    }}

    /* 10. Code blocks and mathematical callouts */
    code {{
        background: {BG_ELEVATED} !important;
        color: {ACCENT_SAFFRON} !important;
        border: 1px solid {BORDER_BASE};
        padding: 2px 5px !important;
        border-radius: 4px !important;
        font-size: 12px !important;
    }}

    /* 11. Footer */
    .footer-text {{
        font-size: 12px;
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER_BASE};
        padding-top: 18px;
        margin-top: 40px;
    }}

    /* 12. Custom Sleek Dark Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BG_BASE};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {BORDER_BASE};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {ACCENT_SAFFRON};
    }}
</style>
"""


def apply_plotly_theme(
    fig: go.Figure,
    height: int = 380,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    """
    Apply unified ISRO-inspired dark/saffron theme to any Plotly figure.
    Ensures crisp contrast, clean tabular typography, and unobtrusive gridlines.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(25, 28, 32, 0.7)",  # BG_SURFACE transparent
        height=height,
        margin=dict(l=35, r=20, t=35 if title else 25, b=35),
        font=dict(
            family=FONT_FAMILY,
            color=TEXT_MUTED,
            size=12,
        ),
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(color=TEXT_WHITE, size=13),
            x=0.01,
            y=0.98,
        ) if title else None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=TEXT_OFFWHITE),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            title=dict(text=xaxis_title, font=dict(color=TEXT_MUTED, size=11)) if xaxis_title else None,
            tickfont=dict(color=TEXT_MUTED, size=11),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            title=dict(text=yaxis_title, font=dict(color=TEXT_MUTED, size=11)) if yaxis_title else None,
            tickfont=dict(color=TEXT_MUTED, size=11),
            zeroline=False,
        ),
    )
    return fig


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    unit: str = "",
    unit_color: str = ACCENT_SAFFRON,
) -> str:
    """Generate HTML snippet for a scientific KPI card."""
    unit_span = (
        f' <span style="font-size: 14px; font-weight: 600; color: {unit_color};">{unit}</span>'
        if unit
        else ""
    )
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}{unit_span}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def meta_card(
    title: str,
    content: str,
    border_left_color: Optional[str] = None,
) -> str:
    """Generate HTML snippet for an information/provenance card."""
    border_style = (
        f'border-left: 3px solid {border_left_color};'
        if border_left_color
        else ""
    )
    return f"""
    <div class="meta-card" style="{border_style}">
        <div style="font-weight: 700; color: {TEXT_WHITE}; margin-bottom: 6px; font-size: 13px;">{title}</div>
        <div style="font-size: 12px; color: {TEXT_MUTED}; line-height: 1.5;">{content}</div>
    </div>
    """
