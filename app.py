"""
Loneliness Risk Index Dashboard
Based on Lokshin & Foster's Loneliness Risk Index methodology

Uses UN World Population Prospects 2024 data.
Run with: streamlit run app.py
"""

from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Loneliness Risk Index Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME & CONSTANTS
# ============================================================================

# All theme colors in one place
THEME = {
    'bg': '#dce8dc',           # Sage page background
    'card': '#ffffff',         # White cards
    'border': '#c8d8c8',      # Card borders
    'border_dark': '#b8ccb8',  # Metric card borders
    'accent': '#3a7d5c',      # Forest green accent
    'accent_dark': '#2d5a4a',  # Darker green (hover/gradient)
    'text': '#2d3436',         # Primary text
    'text_muted': '#495057',   # Secondary text
}

COLOR_PALETTE = {
    'primary': '#3a7d5c',      # Primary country (green)
    'secondary': '#d4a843',    # Comparator country (gold)
    'male': '#3498db',
    'female': '#e91e63',
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'purple': '#9b59b6',
}

LAYOUT_DEFAULTS = {
    'plot_bgcolor': 'white',
    'font_size_title': 16,
    'font_size_axis': 14,
    'font_size_tick': 12,
    'grid_color': 'lightgray',
}

# External links
UN_WPP_URL = "https://population.un.org/wpp/"
UN_WPP_DOWNLOAD_URL = "https://population.un.org/wpp/Download/Standard/CSV/"

# Country name mapping: UN WPP names -> Plotly/ISO names (for world map)
COUNTRY_NAME_MAP = {
    'Türkiye': 'Turkey',
    'United States of America': 'United States',
    'Russian Federation': 'Russia',
    'Iran (Islamic Republic of)': 'Iran',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Republic of Korea': 'South Korea',
    "Dem. People's Republic of Korea": 'North Korea',
    'United Republic of Tanzania': 'Tanzania',
    'Viet Nam': 'Vietnam',
    "Lao People's Democratic Republic": 'Laos',
    'Syrian Arab Republic': 'Syria',
    'Democratic Republic of the Congo': 'Democratic Republic of the Congo',
    'Congo': 'Republic of the Congo',
    "Côte d'Ivoire": 'Ivory Coast',
    'Czechia': 'Czech Republic',
    'State of Palestine': 'Palestine',
    'Republic of Moldova': 'Moldova',
    'North Macedonia': 'North Macedonia',
    'Brunei Darussalam': 'Brunei',
    'Cabo Verde': 'Cape Verde',
    'Timor-Leste': 'East Timor',
    'Eswatini': 'Eswatini',
    'Micronesia (Fed. States of)': 'Micronesia',
}

# ============================================================================
# CSS
# ============================================================================

CUSTOM_CSS = f"""
<style>
    /* Page background */
    .stApp {{
        background-color: {THEME['bg']};
    }}
    [data-testid="stHeader"] {{
        display: none;
    }}
    .block-container {{
        padding-top: 0.5rem !important;
    }}

    /* Dashboard header */
    .main-header {{
        background: linear-gradient(135deg, {THEME['accent']} 0%, {THEME['accent_dark']} 100%);
        padding: 0.6rem 1.8rem;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        color: white;
        position: sticky;
        top: 0;
        z-index: 999;
    }}
    .main-header h1 {{
        color: white !important;
        margin-bottom: 0.1rem;
        font-size: 1.3rem;
    }}
    .main-header p {{
        color: rgba(255,255,255,0.9);
        font-size: 0.8rem;
        margin-bottom: 0;
    }}

    /* Intro text */
    .intro-text {{
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        color: {THEME['text']};
    }}
    .intro-text a {{
        color: {THEME['accent']};
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {THEME['card']};
        border: 1px solid {THEME['border_dark']};
        border-radius: 10px;
        padding: 0.5rem 0.7rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }}
    div[data-testid="stMetric"] label {{
        color: {THEME['text_muted']};
        font-weight: 500;
        font-size: 0.78rem;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        color: {THEME['text']};
    }}

    /* Metrics row labels */
    .metrics-primary {{
        background-color: rgba(58, 125, 92, 0.08);
        border-left: 3px solid {COLOR_PALETTE['primary']};
        padding: 0.25rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.2rem;
    }}
    .metrics-comparator {{
        background-color: rgba(212, 168, 67, 0.1);
        border-left: 3px solid {COLOR_PALETTE['secondary']};
        padding: 0.25rem 0.8rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.2rem;
    }}
    .metrics-primary h3, .metrics-comparator h3 {{
        margin: 0;
        font-size: 0.9rem;
    }}

    /* Spacing */
    .main .stMarkdown h2 {{
        font-size: 1.2rem;
        margin-top: 0.3rem;
        margin-bottom: 0.3rem;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {THEME['card']};
        border-radius: 10px 10px 0 0;
        padding: 6px 14px;
        border: 1px solid {THEME['border']};
        font-size: 0.85rem;
        color: {THEME['text']};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {THEME['accent']};
        color: white;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {THEME['card']};
        border-radius: 0 0 10px 10px;
        padding: 1rem;
        border: 1px solid {THEME['border']};
        border-top: none;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {THEME['bg']};
    }}
    section[data-testid="stSidebar"] .stMarkdown strong {{
        color: {THEME['accent']};
    }}

    /* Footer */
    .footer {{
        background-color: {THEME['card']};
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        border: 1px solid {THEME['border']};
    }}
    .footer a {{
        color: {THEME['accent']};
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}

    /* Download buttons */
    .stDownloadButton button {{
        background-color: {THEME['accent']};
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }}
    .stDownloadButton button:hover {{
        background-color: {THEME['accent_dark']};
    }}
</style>
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def color_dot(color: str) -> str:
    """Colored circle indicator matching a graph line color."""
    return (f'<span style="display:inline-block; width:10px; height:10px; '
            f'border-radius:50%; background:{color}; margin-right:6px; '
            f'vertical-align:middle;"></span>')


def sidebar_label(text: str) -> None:
    """Render a styled sidebar section label."""
    st.sidebar.markdown(
        f'<p style="color:{THEME["accent"]}; font-size:1rem; '
        f'font-weight:700; margin:0.8rem 0 0.3rem;">{text}</p>',
        unsafe_allow_html=True
    )


def filter_elderly(df: pd.DataFrame, T: int, c_max: int) -> pd.DataFrame:
    """Filter dataframe to elderly ages (>= T and <= c_max)."""
    return df[(df['AgeGrpStart'] >= T) & (df['AgeGrpStart'] <= c_max)].copy()


def vulnerability_factor(ages: np.ndarray, T: int, alpha: float) -> np.ndarray:
    """V_c(a) = ((c - T + 1) / (T + 1))^alpha."""
    return np.power((ages - T + 1) / (T + 1), alpha)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                fill_value: float = 1.0) -> np.ndarray:
    """Division with zero-handling."""
    safe_denom = np.where(denominator == 0, fill_value, denominator)
    return numerator / safe_denom


# ============================================================================
# PLOTLY LAYOUT HELPERS
# ============================================================================

def make_title(text: str, size: int = 16) -> dict:
    return dict(text=text, font=dict(size=size))


def make_xaxis(ages: np.ndarray, title: str = 'Age cohort') -> dict:
    return dict(
        title=dict(text=title, font=dict(size=LAYOUT_DEFAULTS['font_size_axis'])),
        tickfont=dict(size=LAYOUT_DEFAULTS['font_size_tick']),
        range=[ages.min() - 1, ages.max() + 1],
        dtick=5, showgrid=True, gridcolor=LAYOUT_DEFAULTS['grid_color']
    )


def make_yaxis(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=LAYOUT_DEFAULTS['font_size_axis'])),
        tickfont=dict(size=LAYOUT_DEFAULTS['font_size_tick']),
        showgrid=True, gridcolor=LAYOUT_DEFAULTS['grid_color']
    )


def make_legend(position: str = 'top-right') -> dict:
    positions = {
        'top-right': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'right', 'x': 0.99},
        'top-left': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'left', 'x': 0.01},
    }
    return dict(**positions.get(position, positions['top-right']),
                bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1)


def add_comparison_traces(fig: go.Figure, primary_data: dict, comparator_data: dict,
                          primary_loc: str, comparator_loc: str,
                          x_key: str, y_key: str,
                          scale: float = 1.0, fill_primary: bool = False) -> None:
    """Add primary + comparator traces using consistent colors."""
    kwargs = dict(
        x=primary_data[x_key], y=primary_data[y_key] * scale,
        mode='lines', name=primary_loc,
        line=dict(color=COLOR_PALETTE['primary'], width=2.5),
    )
    if fill_primary:
        kwargs['fill'] = 'tozeroy'
        kwargs['fillcolor'] = 'rgba(58, 125, 92, 0.1)'
    fig.add_trace(go.Scatter(**kwargs))

    fig.add_trace(go.Scatter(
        x=comparator_data[x_key], y=comparator_data[y_key] * scale,
        mode='lines', name=comparator_loc,
        line=dict(color=COLOR_PALETTE['secondary'], width=2.5),
    ))


# ============================================================================
# LONELINESS INDEX CALCULATIONS
# ============================================================================

def calculate_loneliness_indices(df_country_year: pd.DataFrame, T: int,
                                  alpha: float, c_max: int) -> Optional[Dict[str, Any]]:
    """
    Calculate Loneliness Index components for a specific country-year.

    LI_c = |g_c| x V_c(a) x s_c
    LII  = sum(LI_c) x 100
    LBI  = S_T x LII
    """
    elderly_df = filter_elderly(df_country_year, T, c_max)
    if elderly_df.empty:
        return None

    elderly_df = elderly_df.sort_values('AgeGrpStart')
    ages = elderly_df['AgeGrpStart'].values
    pop_male = elderly_df['PopMale'].values
    pop_female = elderly_df['PopFemale'].values

    N_T = np.sum(pop_male + pop_female)
    if N_T == 0:
        return None

    total_pop = df_country_year['PopTotal'].sum()
    S_T = N_T / total_pop if total_pop > 0 else 0
    s_c = (pop_male + pop_female) / N_T
    g_c = safe_divide(pop_female - pop_male, pop_female + pop_male)
    V_c = vulnerability_factor(ages, T, alpha)
    LI_c = np.abs(g_c) * V_c * s_c
    LII = np.sum(LI_c) * 100
    LBI = S_T * np.sum(LI_c) * 100
    MF_ratio = np.sum(pop_male) / np.sum(pop_female) if np.sum(pop_female) > 0 else 0

    return {
        'ages': ages, 'pop_male': pop_male, 'pop_female': pop_female,
        's_c': s_c, 'g_c': g_c, 'V_c': V_c, 'LI_c': LI_c,
        'LII': LII, 'LBI': LBI, 'S_T': S_T, 'N_T': N_T, 'MF_ratio': MF_ratio,
    }


def calculate_time_series_fast(df_location: pd.DataFrame, location: str,
                                T: int, alpha: float, c_max: int) -> pd.DataFrame:
    """Calculate LII and LBI for all years using vectorized operations."""
    elderly_df = filter_elderly(df_location, T, c_max)
    if elderly_df.empty:
        return pd.DataFrame()

    ages = np.arange(T, c_max + 1)
    V_c = vulnerability_factor(ages, T, alpha)
    elderly_df['V_c'] = elderly_df['AgeGrpStart'].map(dict(zip(ages, V_c)))

    pop_total = (elderly_df['PopMale'] + elderly_df['PopFemale']).replace(0, 1)
    elderly_df['g_c'] = (elderly_df['PopFemale'] - elderly_df['PopMale']) / pop_total
    elderly_df['pop_cohort'] = elderly_df['PopMale'] + elderly_df['PopFemale']
    elderly_df['s_c'] = elderly_df['pop_cohort'] / elderly_df.groupby('Time')['pop_cohort'].transform('sum')
    elderly_df['LI_c'] = np.abs(elderly_df['g_c']) * elderly_df['V_c'] * elderly_df['s_c']

    yearly = elderly_df.groupby('Time').agg(
        PopMale=('PopMale', 'sum'), PopFemale=('PopFemale', 'sum'), LI_c=('LI_c', 'sum')
    ).reset_index()

    yearly['N_T'] = yearly['PopMale'] + yearly['PopFemale']
    yearly['MF_ratio'] = yearly['PopMale'] / yearly['PopFemale'].replace(0, 1)

    total_pop = df_location.groupby('Time')['PopTotal'].sum().reset_index()
    total_pop.columns = ['Time', 'TotalPop']
    yearly = yearly.merge(total_pop, on='Time')

    yearly['S_T'] = yearly['N_T'] / yearly['TotalPop']
    yearly['LII'] = yearly['LI_c'] * 100
    yearly['LBI'] = yearly['S_T'] * yearly['LI_c'] * 100
    yearly['Location'] = location

    return yearly.rename(columns={'Time': 'Year'})[['Year', 'Location', 'LII', 'LBI', 'S_T', 'MF_ratio']]


@st.cache_data(show_spinner="Calculating global LBI data...")
def calculate_global_lbi(df_hash: int, year: int, T: int, alpha: float,
                         c_max: int, _df: pd.DataFrame) -> pd.DataFrame:
    """Calculate LBI for all locations for a given year."""
    df_year = _df[_df['Time'] == year]
    results = []
    for location in df_year['Location'].unique():
        indices = calculate_loneliness_indices(df_year[df_year['Location'] == location], T, alpha, c_max)
        if indices:
            results.append({k: indices[k] for k in ('LII', 'LBI', 'S_T', 'MF_ratio', 'N_T')} | {'Location': location})
    return pd.DataFrame(results)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner="Loading UN population data...")
def load_un_data() -> Optional[pd.DataFrame]:
    """Load UN World Population Prospects 2024 data."""
    parquet_file = os.path.join(os.path.dirname(__file__), "wpp2024_population.parquet")

    if os.path.exists(parquet_file):
        df = pd.read_parquet(parquet_file)
    else:
        data_path = r"F:\OneDrive\__Documents\Papers\Loneliness Index\Data"
        files = [
            os.path.join(data_path, "WPP2024_Population1JanuaryBySingleAgeSex_Medium_1950-2023.csv"),
            os.path.join(data_path, "WPP2024_Population1JanuaryBySingleAgeSex_Medium_2024-2100.csv"),
        ]
        cols = ['Location', 'Time', 'AgeGrpStart', 'PopMale', 'PopFemale']
        dfs = [pd.read_csv(f, usecols=cols, dtype={'Location': 'category', 'Time': 'int16',
               'AgeGrpStart': 'int8', 'PopMale': 'float32', 'PopFemale': 'float32'})
               for f in files if os.path.exists(f)]
        if not dfs:
            return None
        df = pd.concat(dfs, ignore_index=True).dropna(subset=cols)

    df['PopTotal'] = df['PopMale'] + df['PopFemale']
    return df


@st.cache_data(show_spinner="Calculating time series...")
def get_time_series_cached(df_hash: int, location: str, T: int, alpha: float,
                           c_max: int, _df: pd.DataFrame) -> pd.DataFrame:
    return calculate_time_series_fast(_df[_df['Location'] == location], location, T, alpha, c_max)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_intensity_curves(primary_data, comparator_data, primary_loc, comparator_loc, year):
    """Age-specific Loneliness Intensity curves (LI_c)."""
    fig = go.Figure()
    add_comparison_traces(fig, primary_data, comparator_data,
                          primary_loc, comparator_loc, 'ages', 'LI_c', scale=100.0)
    fig.update_layout(
        title=make_title(f'Age-Specific Loneliness Index LI(c), {year}'),
        xaxis=make_xaxis(primary_data['ages']),
        yaxis=make_yaxis('LI(c) = |g(c)| × V(c) × s(c)'),
        legend=make_legend(), plot_bgcolor='white', hovermode='x unified', height=500
    )
    return fig


def plot_burden_curves(primary_data, comparator_data, primary_loc, comparator_loc, year):
    """Loneliness Burden curves: LB_c = S_T × LI_c."""
    fig = go.Figure()
    primary_burden = {'ages': primary_data['ages'], 'LB_c': primary_data['S_T'] * primary_data['LI_c']}
    comparator_burden = {'ages': comparator_data['ages'], 'LB_c': comparator_data['S_T'] * comparator_data['LI_c']}
    add_comparison_traces(fig, primary_burden, comparator_burden,
                          primary_loc, comparator_loc, 'ages', 'LB_c', scale=100.0, fill_primary=True)
    fig.update_layout(
        title=make_title(f'Age-Specific Loneliness Burden LB(c) = S_T × LI(c), {year}'),
        xaxis=make_xaxis(primary_data['ages']),
        yaxis=make_yaxis('LB(c) = S_T × |g(c)| × V(c) × s(c)'),
        legend=make_legend(), plot_bgcolor='white', hovermode='x unified', height=500
    )
    return fig


def plot_time_series(primary_ts, comparator_ts, metric='LBI'):
    """LII or LBI time series comparison."""
    fig = go.Figure()
    primary_loc = primary_ts['Location'].iloc[0] if not primary_ts.empty else 'Location 1'
    comparator_loc = comparator_ts['Location'].iloc[0] if not comparator_ts.empty else 'Location 2'

    for ts, loc, color in [(primary_ts, primary_loc, COLOR_PALETTE['primary']),
                            (comparator_ts, comparator_loc, COLOR_PALETTE['secondary'])]:
        fig.add_trace(go.Scatter(
            x=ts['Year'], y=ts[metric], mode='lines+markers', name=loc,
            line=dict(color=color, width=2), marker=dict(size=4)
        ))

    fig.add_vline(x=2024, line_dash="dash", line_color="gray",
                  annotation_text="2024", annotation_position="top right")

    title_map = {'LII': 'Loneliness Intensity Index (LII)',
                 'LBI': 'Loneliness Burden Index (LBI)',
                 'S_T': 'Share of Elderly Population (S_T)'}
    fig.update_layout(title=title_map.get(metric, metric) + ' Over Time',
                      xaxis_title='Year', yaxis_title=metric,
                      legend=make_legend('top-left'), hovermode='x unified')
    return fig


def plot_components(data, location, year, alpha):
    """Three components of the index: g_c, V_c, s_c."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        'Gender Ratio g_c', f'Vulnerability V_c(α={alpha})', 'Cohort Share s_c'))

    fig.add_trace(go.Bar(x=data['ages'], y=data['g_c'], name='g_c',
        marker_color=np.where(data['g_c'] >= 0, COLOR_PALETTE['positive'], COLOR_PALETTE['negative'])),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=data['ages'], y=data['V_c'], mode='lines+markers',
        name='V_c(α)', line=dict(color=COLOR_PALETTE['purple'], width=2)),
        row=1, col=2)
    fig.add_trace(go.Bar(x=data['ages'], y=data['s_c'], name='s_c',
        marker_color=COLOR_PALETTE['male']), row=1, col=3)

    fig.update_layout(title=f'Components of Loneliness Index: {location} ({year})',
                      showlegend=False, height=350)
    for col in range(1, 4):
        fig.update_xaxes(title_text='Age', row=1, col=col)
    return fig


def plot_pyramid(data, location, year):
    """Elderly population pyramid."""
    fig = go.Figure()
    max_pop = max(np.max(data['pop_male']), np.max(data['pop_female']))

    fig.add_trace(go.Bar(y=data['ages'], x=-data['pop_male'], orientation='h',
        name='Males', marker_color=COLOR_PALETTE['male'],
        hovertemplate='Age %{y}<br>Males: %{customdata:.1f}k<extra></extra>',
        customdata=data['pop_male']))
    fig.add_trace(go.Bar(y=data['ages'], x=data['pop_female'], orientation='h',
        name='Females', marker_color=COLOR_PALETTE['female'],
        hovertemplate='Age %{y}<br>Females: %{x:.1f}k<extra></extra>'))

    fig.update_layout(
        title=f'Elderly Population Pyramid: {location} ({year})',
        xaxis=dict(title='Population (thousands)', range=[-max_pop * 1.1, max_pop * 1.1],
            tickvals=[-max_pop, -max_pop/2, 0, max_pop/2, max_pop],
            ticktext=[f'{max_pop:.0f}', f'{max_pop/2:.0f}', '0', f'{max_pop/2:.0f}', f'{max_pop:.0f}']),
        yaxis_title='Age', barmode='overlay', bargap=0.1, height=450)
    return fig


def plot_world_map(global_lbi, year, metric='LBI'):
    """Interactive world choropleth map."""
    titles = {'LBI': 'Loneliness Burden Index (LBI)', 'LII': 'Loneliness Intensity Index (LII)',
              'S_T': 'Share of Elderly Population'}
    scales = {'LBI': 'YlOrRd', 'LII': 'YlOrRd', 'S_T': 'Blues'}

    plot_data = global_lbi.copy()
    plot_data['PlotLocation'] = plot_data['Location'].map(lambda x: COUNTRY_NAME_MAP.get(x, x))

    fig = px.choropleth(
        plot_data, locations='PlotLocation', locationmode='country names', color=metric,
        hover_name='Location',
        hover_data={'PlotLocation': False, 'Location': False,
                    'LBI': ':.3f', 'LII': ':.3f', 'S_T': ':.2%', 'MF_ratio': ':.3f'},
        color_continuous_scale=scales.get(metric, 'YlOrRd'),
        title=f'{titles.get(metric, metric)} by Country ({year})')

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor='lightgray',
                 projection_type='natural earth', showland=True, landcolor='#f8f9fa',
                 showocean=True, oceancolor='#e6f2ff', showcountries=True, countrycolor='lightgray'),
        height=600, margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title=metric, tickformat='.2f' if metric != 'S_T' else '.0%'))
    fig.update_geos(visible=True, resolution=110, showcountries=True)
    return fig


# ============================================================================
# UI SECTIONS
# ============================================================================

def render_sidebar(locations, years):
    """Render sidebar controls and return selected parameters."""
    sidebar_label("Locations")
    primary_loc = st.sidebar.selectbox(
        "Primary", options=locations,
        index=locations.index('Japan') if 'Japan' in locations else 0)
    comparator_options = [loc for loc in locations if loc != primary_loc]
    comparator_loc = st.sidebar.selectbox(
        "Comparator", options=comparator_options,
        index=comparator_options.index('Germany') if 'Germany' in comparator_options else 0)

    sidebar_label("Parameters")
    alpha = st.sidebar.slider("α (vulnerability)", 0.5, 2.5, 1.5, 0.1,
                              help="Controls how vulnerability increases with age. Paper uses α=1.5")
    T = st.sidebar.slider("T (elderly threshold)", 55, 70, 60, 5,
                          help="Age threshold for elderly population")
    c_max = st.sidebar.slider("c_max (max age)", 85, 100, 98, 1,
                              help="Maximum age in analysis")

    sidebar_label("Time")
    year = st.sidebar.slider("Cross-section year", int(min(years)), int(max(years)), 2023, 1)
    year_range = st.sidebar.slider("Time series range", int(min(years)), int(max(years)), (1950, 2100), 5)

    with st.sidebar.expander("About", expanded=False):
        st.markdown(f"Based on Lokshin & Foster's **Loneliness Risk Index**. "
                    f"Data: [UN WPP 2024]({UN_WPP_URL})")

    return primary_loc, comparator_loc, alpha, T, c_max, year, year_range


def render_metrics(primary_data, comparator_data, primary_loc, comparator_loc, year, T):
    """Render the summary metrics section."""
    st.markdown("#### Summary Metrics")

    # Primary country
    st.markdown(f'<div class="metrics-primary"><h3>'
                f'{color_dot(COLOR_PALETTE["primary"])}{primary_loc} ({year})</h3></div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    metrics = [
        ("Loneliness Intensity Index", primary_data['LII'], comparator_data['LII'], '.3f'),
        ("Loneliness Burden Index", primary_data['LBI'], comparator_data['LBI'], '.3f'),
        (f"Share of elderly ({T}+)", primary_data['S_T'], comparator_data['S_T'], '.2%'),
        (f"M/F ratio ({T}+)", primary_data['MF_ratio'], comparator_data['MF_ratio'], '.3f'),
    ]
    for col, (label, pval, cval, fmt) in zip(cols, metrics):
        delta = pval - cval
        dfmt = '+.4f' if fmt == '.3f' else ('+.2%' if fmt == '.2%' else '+.3f')
        with col:
            st.metric(label=label, value=f"{pval:{fmt}}", delta=f"{delta:{dfmt}}" if delta != 0 else None)

    # Comparator country
    st.markdown(f'<div class="metrics-comparator"><h3>'
                f'{color_dot(COLOR_PALETTE["secondary"])}{comparator_loc} ({year})</h3></div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (label, _, cval, fmt) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=f"{cval:{fmt}}")


def render_footer(alpha, T, c_max):
    """Render the footer with methodology info."""
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <h4>About</h4>
        <p><strong>Methodology:</strong> Lokshin, M. and J. Foster. "Loneliness Risk Index."</p>
        <p><strong>Data Source:</strong>
            <a href="{UN_WPP_URL}" target="_blank">UN World Population Prospects 2024</a>
            (<a href="{UN_WPP_DOWNLOAD_URL}" target="_blank">Download Data</a>)</p>
        <hr style="border-color: #e9ecef; margin: 1rem 0;">
        <h4>Current Parameters</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 0.3rem 0;"><strong>Vulnerability (α)</strong></td>
                <td style="padding: 0.3rem 0;">{alpha}</td>
                <td style="padding: 0.3rem 0;"><strong>Elderly threshold (T)</strong></td>
                <td style="padding: 0.3rem 0;">{T}</td>
                <td style="padding: 0.3rem 0;"><strong>Max age (c_max)</strong></td>
                <td style="padding: 0.3rem 0;">{c_max}</td>
            </tr>
        </table>
        <hr style="border-color: #e9ecef; margin: 1rem 0;">
        <h4>Formulas</h4>
        <ul style="margin: 0.5rem 0;">
            <li><strong>LI<sub>c</sub></strong> = |g<sub>c</sub>| × V<sub>c</sub>(α) × s<sub>c</sub></li>
            <li><strong>LII</strong> = Σ LI<sub>c</sub> — Loneliness Intensity Index</li>
            <li><strong>LBI</strong> = S<sub>T</sub> × LII — Loneliness Burden Index</li>
        </ul>
        <p style="margin-top: 1rem; font-size: 0.85rem; color: #6c757d;">
            Where: g<sub>c</sub> = normalized gender ratio, V<sub>c</sub> = vulnerability factor,
            s<sub>c</sub> = cohort share, S<sub>T</sub> = elderly population share
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>Loneliness Risk Index Dashboard</h1>
        <p>Demographic loneliness risk visualization based on Lokshin & Foster methodology</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="intro-text">Compare <b>loneliness burden</b> across countries using '
                f'<a href="{UN_WPP_URL}">UN World Population Prospects 2024</a> data.</p>',
                unsafe_allow_html=True)

    df = load_un_data()
    if df is None:
        st.stop()

    locations = sorted(df['Location'].unique())
    years = sorted(df['Time'].unique())

    # Sidebar
    primary_loc, comparator_loc, alpha, T, c_max, year, year_range = render_sidebar(locations, years)

    # Calculate indices
    primary_data = calculate_loneliness_indices(df[(df['Location'] == primary_loc) & (df['Time'] == year)], T, alpha, c_max)
    comparator_data = calculate_loneliness_indices(df[(df['Location'] == comparator_loc) & (df['Time'] == year)], T, alpha, c_max)

    if primary_data is None or comparator_data is None:
        st.error("Could not calculate indices. Check if data exists for selected locations/year.")
        st.stop()

    df_hash = hash(tuple(df['Location'].unique()))
    primary_ts = get_time_series_cached(df_hash, primary_loc, T, alpha, c_max, df)
    comparator_ts = get_time_series_cached(df_hash, comparator_loc, T, alpha, c_max, df)
    primary_ts = primary_ts[(primary_ts['Year'] >= year_range[0]) & (primary_ts['Year'] <= year_range[1])]
    comparator_ts = comparator_ts[(comparator_ts['Year'] >= year_range[0]) & (comparator_ts['Year'] <= year_range[1])]

    # Summary metrics
    render_metrics(primary_data, comparator_data, primary_loc, comparator_loc, year, T)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📉 Time Series", "📊 Age-Specific Curves", "🔬 Components",
        "👥 Population Pyramids", "📋 Data Table", "🗺️ World Map"])

    with tab1:
        st.subheader("Loneliness Indices Over Time")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_time_series(primary_ts, comparator_ts, 'LII'), use_container_width=True)
        with c2:
            st.plotly_chart(plot_time_series(primary_ts, comparator_ts, 'LBI'), use_container_width=True)
        st.plotly_chart(plot_time_series(primary_ts, comparator_ts, 'S_T'), use_container_width=True)

    with tab2:
        st.subheader(f"Age-Specific Analysis ({year})")
        st.plotly_chart(plot_intensity_curves(primary_data, comparator_data, primary_loc, comparator_loc, year), use_container_width=True)
        st.plotly_chart(plot_burden_curves(primary_data, comparator_data, primary_loc, comparator_loc, year), use_container_width=True)

    with tab3:
        st.subheader("Index Components")
        st.markdown(f"**Formula**: LI_c = |g_c| × V_c(α), where: "
                    f"g_c = normalized gender gap, V_c(α) = vulnerability (α={alpha}), s_c = cohort share")
        st.plotly_chart(plot_components(primary_data, primary_loc, year, alpha), use_container_width=True)
        st.plotly_chart(plot_components(comparator_data, comparator_loc, year, alpha), use_container_width=True)

    with tab4:
        st.subheader("Elderly Population Structure")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_pyramid(primary_data, primary_loc, year), use_container_width=True)
        with c2:
            st.plotly_chart(plot_pyramid(comparator_data, comparator_loc, year), use_container_width=True)

    with tab5:
        st.subheader("Detailed Data")
        comparison_df = pd.DataFrame({
            'Age': primary_data['ages'],
            f'{primary_loc} - Males (k)': primary_data['pop_male'],
            f'{primary_loc} - Females (k)': primary_data['pop_female'],
            f'{primary_loc} - g_c': primary_data['g_c'],
            f'{primary_loc} - V_c': primary_data['V_c'],
            f'{primary_loc} - s_c': primary_data['s_c'],
            f'{primary_loc} - LI_c': primary_data['LI_c'],
            f'{comparator_loc} - Males (k)': comparator_data['pop_male'],
            f'{comparator_loc} - Females (k)': comparator_data['pop_female'],
            f'{comparator_loc} - g_c': comparator_data['g_c'],
            f'{comparator_loc} - LI_c': comparator_data['LI_c'],
        })
        st.dataframe(comparison_df.style.format({
            col: '{:.3f}' for col in comparison_df.columns if col != 'Age'
        }), use_container_width=True)

        st.subheader("📥 Download Results")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download Time Series (CSV)",
                pd.concat([primary_ts, comparator_ts]).to_csv(index=False),
                f"loneliness_ts_{primary_loc}_{comparator_loc}.csv", "text/csv")
        with c2:
            cross_df = pd.DataFrame({
                'Age': primary_data['ages'],
                **{f'{primary_loc}_{k}': primary_data[v] for k, v in
                   [('PopMale', 'pop_male'), ('PopFemale', 'pop_female'),
                    ('g_c', 'g_c'), ('V_c', 'V_c'), ('s_c', 's_c'), ('LI_c', 'LI_c')]},
                **{f'{comparator_loc}_{k}': comparator_data[v] for k, v in
                   [('PopMale', 'pop_male'), ('PopFemale', 'pop_female'),
                    ('g_c', 'g_c'), ('LI_c', 'LI_c')]},
            })
            st.download_button("Download Cross-Section (CSV)",
                cross_df.to_csv(index=False),
                f"loneliness_cross_{primary_loc}_{comparator_loc}_{year}.csv", "text/csv")

    with tab6:
        st.subheader(f"Global Loneliness Burden Index ({year})")
        global_lbi = calculate_global_lbi(df_hash, year, T, alpha, c_max, df)
        map_metric = st.radio("Select metric:", ['LBI', 'LII', 'S_T'],
            format_func=lambda x: {'LBI': 'Loneliness Burden Index',
                                   'LII': 'Loneliness Intensity Index',
                                   'S_T': 'Share of Elderly Population'}[x],
            horizontal=True)
        st.plotly_chart(plot_world_map(global_lbi, year, map_metric), use_container_width=True, config={'scrollZoom': True})

        c1, c2 = st.columns(2)
        fmt = {'LBI': '{:.3f}', 'LII': '{:.3f}', 'S_T': '{:.2%}'}
        for col, label, sorter in [(c1, "Top 10", 'nlargest'), (c2, "Bottom 10", 'nsmallest')]:
            with col:
                st.markdown(f"**{label} Countries by {map_metric}**")
                data = getattr(global_lbi, sorter)(10, map_metric)[['Location', 'LBI', 'LII', 'S_T']].reset_index(drop=True)
                data.index = data.index + 1
                st.dataframe(data.style.format(fmt), use_container_width=True)

    render_footer(alpha, T, c_max)


if __name__ == "__main__":
    main()
