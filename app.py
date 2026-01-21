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
# CONSTANTS
# ============================================================================
DATA_PATH = r"F:\OneDrive\__Documents\Papers\Age_FGT\Data"

# External links
UN_WPP_URL = "https://population.un.org/wpp/"
UN_WPP_DOWNLOAD_URL = "https://population.un.org/wpp/Download/Standard/CSV/"

# Country name mapping: UN WPP names -> Plotly/ISO names
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

# Custom CSS for styling
CUSTOM_CSS = """
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }

    /* Metric cards styling */
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label {
        color: #495057;
        font-weight: 500;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #495057;
    }

    /* Info box styling */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Footer styling */
    .footer {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        border: 1px solid #e9ecef;
    }
    .footer a {
        color: #667eea;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }

    /* Download buttons */
    .stDownloadButton button {
        background-color: #667eea;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stDownloadButton button:hover {
        background-color: #5a6fd6;
    }
</style>
"""

COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'black': '#000000',
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def filter_elderly_population(df: pd.DataFrame, T: int, c_max: int) -> pd.DataFrame:
    """Filter dataframe to elderly ages (>= T and <= c_max)."""
    return df[(df['AgeGrpStart'] >= T) & (df['AgeGrpStart'] <= c_max)].copy()


def calculate_vulnerability_factor(ages: np.ndarray, T: int, alpha: float) -> np.ndarray:
    """Calculate vulnerability factor V_c(a) = ((c - T + 1) / (T + 1))^alpha."""
    return np.power((ages - T + 1) / (T + 1), alpha)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                fill_value: float = 1.0) -> np.ndarray:
    """Division with zero-handling, replacing zeros in denominator with fill_value."""
    safe_denom = np.where(denominator == 0, fill_value, denominator)
    return numerator / safe_denom


# ============================================================================
# LAYOUT HELPER FUNCTIONS
# ============================================================================

def create_title(text: str, size: int = 16) -> dict:
    """Create standardized Plotly title configuration."""
    return dict(text=text, font=dict(size=size))


def create_standard_xaxis(ages: np.ndarray, title: str = 'Age cohort') -> dict:
    """Create standardized x-axis configuration for age-based plots."""
    return dict(
        title=dict(text=title, font=dict(size=LAYOUT_DEFAULTS['font_size_axis'])),
        tickfont=dict(size=LAYOUT_DEFAULTS['font_size_tick']),
        range=[ages.min() - 1, ages.max() + 1],
        dtick=5,
        showgrid=True,
        gridcolor=LAYOUT_DEFAULTS['grid_color']
    )


def create_standard_yaxis(title: str) -> dict:
    """Create standardized y-axis configuration."""
    return dict(
        title=dict(text=title, font=dict(size=LAYOUT_DEFAULTS['font_size_axis'])),
        tickfont=dict(size=LAYOUT_DEFAULTS['font_size_tick']),
        showgrid=True,
        gridcolor=LAYOUT_DEFAULTS['grid_color']
    )


def create_standard_legend(position: str = 'top-right') -> dict:
    """Create standardized legend configuration."""
    positions = {
        'top-right': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'right', 'x': 0.99},
        'top-left': {'yanchor': 'top', 'y': 0.99, 'xanchor': 'left', 'x': 0.01},
    }
    pos = positions.get(position, positions['top-right'])
    return dict(
        **pos,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=1
    )


def add_comparison_traces(fig: go.Figure, primary_data: dict, comparator_data: dict,
                          primary_loc: str, comparator_loc: str,
                          x_key: str, y_key: str,
                          scale: float = 1.0, dash_second: bool = True,
                          fill_primary: bool = False) -> go.Figure:
    """Add two traces for location comparison."""
    primary_y = primary_data[y_key] * scale
    comparator_y = comparator_data[y_key] * scale

    # Primary location - solid line
    trace_kwargs = dict(
        x=primary_data[x_key],
        y=primary_y,
        mode='lines',
        name=primary_loc,
        line=dict(color=COLOR_PALETTE['black'], width=2.5),
    )
    if fill_primary:
        trace_kwargs['fill'] = 'tozeroy'
        trace_kwargs['fillcolor'] = 'rgba(0, 0, 0, 0.1)'
    fig.add_trace(go.Scatter(**trace_kwargs))

    # Comparator location - dashed line
    fig.add_trace(go.Scatter(
        x=comparator_data[x_key],
        y=comparator_y,
        mode='lines',
        name=comparator_loc,
        line=dict(color=COLOR_PALETTE['black'], width=2.5,
                  dash='dash' if dash_second else None),
    ))

    return fig


# ============================================================================
# LONELINESS INDEX CALCULATIONS (from Lokshin & Foster paper)
# ============================================================================

def calculate_loneliness_indices(df_country_year: pd.DataFrame, T: int,
                                  alpha: float, c_max: int) -> Optional[Dict[str, Any]]:
    """
    Calculate Loneliness Index components for a specific country-year.

    From equation (1) in the paper:
    LI_c = |g_c| × V_c(α)

    Where:
    - g_c = (F_c - M_c) / (F_c + M_c)  [normalized gender ratio]
    - V_c(α) = ((c - T + 1) / (T + 1))^α  [vulnerability factor]

    LII = Σ s_c × LI_c  [equation 2]
    LBI = S_T × LII  [equation 3]
    """
    # Filter for elderly ages
    elderly_df = filter_elderly_population(df_country_year, T, c_max)

    if elderly_df.empty:
        return None

    elderly_df = elderly_df.sort_values('AgeGrpStart')

    ages = elderly_df['AgeGrpStart'].values
    pop_male = elderly_df['PopMale'].values  # Already in thousands
    pop_female = elderly_df['PopFemale'].values

    # Total elderly population N_T
    N_T = np.sum(pop_male + pop_female)

    if N_T == 0:
        return None

    # Total population (all ages) for S_T calculation
    total_pop = df_country_year['PopTotal'].sum()

    # Share of elderly in total population: S_T = N_T / N
    S_T = N_T / total_pop if total_pop > 0 else 0

    # Cohort shares: s_c = (M_c + F_c) / N_T
    s_c = (pop_male + pop_female) / N_T

    # Normalized gender ratio: g_c = (F_c - M_c) / (F_c + M_c)
    pop_total_cohort = pop_female + pop_male
    g_c = safe_divide(pop_female - pop_male, pop_total_cohort, fill_value=1.0)

    # Vulnerability factor: V_c(α) = ((c - T + 1) / (T + 1))^α
    V_c = calculate_vulnerability_factor(ages, T, alpha)

    # Age-specific Loneliness Index: LI_c = |g_c| × V_c(α) × s_c
    # (includes cohort share as per Lokshin & Foster)
    LI_c = np.abs(g_c) * V_c * s_c

    # Loneliness Intensity Index: LII = Σ LI_c (multiplied by 100 for display)
    LII = np.sum(LI_c) * 100

    # Loneliness Burden Index: LBI = S_T × LII (multiplied by 100 for display)
    LBI = S_T * np.sum(LI_c) * 100

    # For component plots, also compute raw |g_c| × V_c (without s_c)
    LI_c_raw = np.abs(g_c) * V_c  # This is |g_c| × V_c without cohort weight

    # Male-to-female ratio
    MF_ratio = np.sum(pop_male) / np.sum(pop_female) if np.sum(pop_female) > 0 else 0

    return {
        'ages': ages,
        'pop_male': pop_male,
        'pop_female': pop_female,
        's_c': s_c,
        'g_c': g_c,
        'V_c': V_c,
        'LI_c': LI_c,           # |g_c| × V_c × s_c (the full age-specific index)
        'LI_c_raw': LI_c_raw,   # |g_c| × V_c (without cohort weight, for components)
        'LII': LII,
        'LBI': LBI,
        'S_T': S_T,
        'N_T': N_T,
        'MF_ratio': MF_ratio
    }


def calculate_time_series_fast(df_location: pd.DataFrame, location: str,
                                T: int, alpha: float, c_max: int) -> pd.DataFrame:
    """Calculate LII and LBI for all years using vectorized operations."""

    # Filter for elderly ages once
    elderly_df = filter_elderly_population(df_location, T, c_max)

    if elderly_df.empty:
        return pd.DataFrame()

    # Pre-calculate vulnerability factor (same for all years)
    ages = np.arange(T, c_max + 1)
    V_c = calculate_vulnerability_factor(ages, T, alpha)
    V_c_dict = dict(zip(ages, V_c))
    elderly_df['V_c'] = elderly_df['AgeGrpStart'].map(V_c_dict)

    # Calculate gender gap
    pop_total = elderly_df['PopMale'] + elderly_df['PopFemale']
    pop_total = pop_total.replace(0, 1)  # Avoid division by zero
    elderly_df['g_c'] = (elderly_df['PopFemale'] - elderly_df['PopMale']) / pop_total

    # Calculate s_c (cohort share) - need N_T per year first
    elderly_df['pop_cohort'] = elderly_df['PopMale'] + elderly_df['PopFemale']
    yearly_N_T = elderly_df.groupby('Time')['pop_cohort'].transform('sum')
    elderly_df['s_c'] = elderly_df['pop_cohort'] / yearly_N_T

    # LI_c = |g_c| * V_c * s_c (correct formula)
    elderly_df['LI_c'] = np.abs(elderly_df['g_c']) * elderly_df['V_c'] * elderly_df['s_c']

    # Group by year for aggregation
    yearly = elderly_df.groupby('Time').agg({
        'PopMale': 'sum',
        'PopFemale': 'sum',
        'LI_c': 'sum'  # Sum of LI_c = LII (before *100)
    }).reset_index()

    # Calculate N_T (elderly population) per year
    yearly['N_T'] = yearly['PopMale'] + yearly['PopFemale']
    yearly['MF_ratio'] = yearly['PopMale'] / yearly['PopFemale'].replace(0, 1)

    # Get total population per year from original data
    total_pop = df_location.groupby('Time')['PopTotal'].sum().reset_index()
    total_pop.columns = ['Time', 'TotalPop']
    yearly = yearly.merge(total_pop, on='Time')

    # S_T = N_T / Total Population
    yearly['S_T'] = yearly['N_T'] / yearly['TotalPop']

    # LII = sum of LI_c, LBI = S_T * LII (both * 100 for display)
    yearly['LII'] = yearly['LI_c'] * 100
    yearly['LBI'] = yearly['S_T'] * yearly['LI_c'] * 100

    yearly['Location'] = location
    yearly = yearly.rename(columns={'Time': 'Year'})

    return yearly[['Year', 'Location', 'LII', 'LBI', 'S_T', 'MF_ratio']]


@st.cache_data(show_spinner="Calculating global LBI data...")
def calculate_global_lbi(df_hash: int, year: int, T: int, alpha: float,
                         c_max: int, _df: pd.DataFrame) -> pd.DataFrame:
    """Calculate LBI for all locations for a given year."""
    df_year = _df[_df['Time'] == year]
    locations = df_year['Location'].unique()

    results = []
    for location in locations:
        df_loc = df_year[df_year['Location'] == location]
        indices = calculate_loneliness_indices(df_loc, T, alpha, c_max)
        if indices is not None:
            results.append({
                'Location': location,
                'LBI': indices['LBI'],
                'LII': indices['LII'],
                'S_T': indices['S_T'],
                'MF_ratio': indices['MF_ratio'],
                'N_T': indices['N_T']
            })

    return pd.DataFrame(results)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner="Loading UN population data...")
def load_un_data() -> Optional[pd.DataFrame]:
    """Load UN World Population Prospects 2024 data from parquet file."""

    # Try parquet file first (for deployment), then fall back to CSV (for local dev)
    parquet_file = os.path.join(os.path.dirname(__file__), "wpp2024_population.parquet")

    if os.path.exists(parquet_file):
        # Load from compressed parquet (fast, small file)
        population_df = pd.read_parquet(parquet_file)
    else:
        # Fallback: load from original CSV files
        file1 = os.path.join(DATA_PATH, "WPP2024_Population1JanuaryBySingleAgeSex_Medium_1950-2023.csv")
        file2 = os.path.join(DATA_PATH, "WPP2024_Population1JanuaryBySingleAgeSex_Medium_2024-2100.csv")

        dataframes = []
        cols = ['Location', 'Time', 'AgeGrpStart', 'PopMale', 'PopFemale']

        for filepath in [file1, file2]:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, usecols=cols,
                               dtype={'Location': 'category', 'Time': 'int16',
                                      'AgeGrpStart': 'int8', 'PopMale': 'float32',
                                      'PopFemale': 'float32'})
                dataframes.append(df)

        if not dataframes:
            return None

        population_df = pd.concat(dataframes, ignore_index=True)
        population_df = population_df.dropna(subset=['Location', 'Time', 'AgeGrpStart', 'PopMale', 'PopFemale'])

    # Compute PopTotal on-the-fly
    population_df['PopTotal'] = population_df['PopMale'] + population_df['PopFemale']

    return population_df


@st.cache_data(show_spinner="Calculating time series...")
def get_time_series_cached(df_hash: int, location: str, T: int, alpha: float,
                           c_max: int, _df: pd.DataFrame) -> pd.DataFrame:
    """Cached time series calculation for a location."""
    df_loc = _df[_df['Location'] == location]
    return calculate_time_series_fast(df_loc, location, T, alpha, c_max)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_loneliness_intensity_curves(primary_data: Dict[str, Any], comparator_data: Dict[str, Any],
                                      primary_loc: str, comparator_loc: str, year: int) -> go.Figure:
    """Plot age-specific Loneliness Intensity curves (LI_c) - Figure 2 style."""
    fig = go.Figure()

    add_comparison_traces(fig, primary_data, comparator_data,
                          primary_loc, comparator_loc,
                          x_key='ages', y_key='LI_c', scale=100.0)

    fig.update_layout(
        title=create_title(f'Age-Specific Loneliness Index LI(c), {year}'),
        xaxis=create_standard_xaxis(primary_data['ages']),
        yaxis=create_standard_yaxis('LI(c) = |g(c)| × V(c) × s(c)'),
        legend=create_standard_legend('top-right'),
        plot_bgcolor=LAYOUT_DEFAULTS['plot_bgcolor'],
        hovermode='x unified',
        height=500
    )

    return fig


def plot_loneliness_burden_curves(primary_data: Dict[str, Any], comparator_data: Dict[str, Any],
                                   primary_loc: str, comparator_loc: str, year: int) -> go.Figure:
    """Plot Loneliness Burden curves: LB_c = S_T × LI_c (age-specific contribution to LBI)."""
    fig = go.Figure()

    # Create temporary data with pre-computed LB_c for the helper function
    primary_burden = {'ages': primary_data['ages'],
                      'LB_c': primary_data['S_T'] * primary_data['LI_c']}
    comparator_burden = {'ages': comparator_data['ages'],
                         'LB_c': comparator_data['S_T'] * comparator_data['LI_c']}

    add_comparison_traces(fig, primary_burden, comparator_burden,
                          primary_loc, comparator_loc,
                          x_key='ages', y_key='LB_c', scale=100.0,
                          fill_primary=True)

    fig.update_layout(
        title=create_title(f'Age-Specific Loneliness Burden LB(c) = S_T × LI(c), {year}'),
        xaxis=create_standard_xaxis(primary_data['ages']),
        yaxis=create_standard_yaxis('LB(c) = S_T × |g(c)| × V(c) × s(c)'),
        legend=create_standard_legend('top-right'),
        plot_bgcolor=LAYOUT_DEFAULTS['plot_bgcolor'],
        hovermode='x unified',
        height=500
    )

    return fig


def plot_time_series_comparison(primary_ts: pd.DataFrame, comparator_ts: pd.DataFrame,
                                 metric: str = 'LBI') -> go.Figure:
    """Plot LII or LBI time series - Figure 3 and 5 style."""
    fig = go.Figure()

    primary_loc = primary_ts['Location'].iloc[0] if not primary_ts.empty else 'Location 1'
    comparator_loc = comparator_ts['Location'].iloc[0] if not comparator_ts.empty else 'Location 2'

    # Primary location
    fig.add_trace(go.Scatter(
        x=primary_ts['Year'],
        y=primary_ts[metric],
        mode='lines+markers',
        name=primary_loc,
        line=dict(color=COLOR_PALETTE['primary'], width=2),
        marker=dict(size=4)
    ))

    # Comparator location
    fig.add_trace(go.Scatter(
        x=comparator_ts['Year'],
        y=comparator_ts[metric],
        mode='lines+markers',
        name=comparator_loc,
        line=dict(color=COLOR_PALETTE['secondary'], width=2),
        marker=dict(size=4)
    ))

    # Vertical line at 2024 (actual vs projected)
    fig.add_vline(x=2024, line_dash="dash", line_color="gray",
                  annotation_text="2024", annotation_position="top right")

    title_map = {
        'LII': 'Loneliness Intensity Index (LII)',
        'LBI': 'Loneliness Burden Index (LBI)',
        'S_T': 'Share of Elderly Population (S_T)'
    }

    fig.update_layout(
        title=f'{title_map.get(metric, metric)} Over Time',
        xaxis_title='Year',
        yaxis_title=metric,
        legend=create_standard_legend('top-left'),
        hovermode='x unified'
    )

    return fig


def plot_components(data: Dict[str, Any], location: str, year: int, alpha: float) -> go.Figure:
    """Plot the three components of the index - Figure 1 left panel style."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Gender Ratio g_c',
            f'Vulnerability V_c(α={alpha})',
            'Cohort Share s_c'
        )
    )

    # Gender ratio (can be negative, but we show raw g_c)
    fig.add_trace(
        go.Bar(x=data['ages'], y=data['g_c'], name='g_c',
               marker_color=np.where(data['g_c'] >= 0,
                                     COLOR_PALETTE['positive'],
                                     COLOR_PALETTE['negative'])),
        row=1, col=1
    )

    # Vulnerability factor
    fig.add_trace(
        go.Scatter(x=data['ages'], y=data['V_c'], mode='lines+markers',
                   name='V_c(α)', line=dict(color=COLOR_PALETTE['purple'], width=2)),
        row=1, col=2
    )

    # Cohort share
    fig.add_trace(
        go.Bar(x=data['ages'], y=data['s_c'], name='s_c',
               marker_color=COLOR_PALETTE['male']),
        row=1, col=3
    )

    fig.update_layout(
        title=f'Components of Loneliness Index: {location} ({year})',
        showlegend=False,
        height=350
    )

    fig.update_xaxes(title_text='Age', row=1, col=1)
    fig.update_xaxes(title_text='Age', row=1, col=2)
    fig.update_xaxes(title_text='Age', row=1, col=3)

    return fig


def plot_population_pyramid(data: Dict[str, Any], location: str, year: int) -> go.Figure:
    """Plot elderly population pyramid."""
    fig = go.Figure()

    max_pop = max(np.max(data['pop_male']), np.max(data['pop_female']))

    # Males (left side - negative)
    fig.add_trace(go.Bar(
        y=data['ages'],
        x=-data['pop_male'],
        orientation='h',
        name='Males',
        marker_color=COLOR_PALETTE['male'],
        hovertemplate='Age %{y}<br>Males: %{customdata:.1f}k<extra></extra>',
        customdata=data['pop_male']
    ))

    # Females (right side)
    fig.add_trace(go.Bar(
        y=data['ages'],
        x=data['pop_female'],
        orientation='h',
        name='Females',
        marker_color=COLOR_PALETTE['female'],
        hovertemplate='Age %{y}<br>Females: %{x:.1f}k<extra></extra>'
    ))

    fig.update_layout(
        title=f'Elderly Population Pyramid: {location} ({year})',
        xaxis=dict(
            title='Population (thousands)',
            range=[-max_pop * 1.1, max_pop * 1.1],
            tickvals=[-max_pop, -max_pop/2, 0, max_pop/2, max_pop],
            ticktext=[f'{max_pop:.0f}', f'{max_pop/2:.0f}', '0', f'{max_pop/2:.0f}', f'{max_pop:.0f}']
        ),
        yaxis_title='Age',
        barmode='overlay',
        bargap=0.1,
        height=450
    )

    return fig


def plot_world_map(global_lbi: pd.DataFrame, year: int, metric: str = 'LBI') -> go.Figure:
    """Plot interactive world choropleth map of LBI or LII."""

    metric_titles = {
        'LBI': 'Loneliness Burden Index (LBI)',
        'LII': 'Loneliness Intensity Index (LII)',
        'S_T': 'Share of Elderly Population'
    }

    # Color scales for different metrics
    color_scales = {
        'LBI': 'YlOrRd',
        'LII': 'YlOrRd',
        'S_T': 'Blues'
    }

    # Map UN country names to Plotly-compatible names
    plot_data = global_lbi.copy()
    plot_data['PlotLocation'] = plot_data['Location'].map(
        lambda x: COUNTRY_NAME_MAP.get(x, x)
    )

    fig = px.choropleth(
        plot_data,
        locations='PlotLocation',
        locationmode='country names',
        color=metric,
        hover_name='Location',
        hover_data={
            'PlotLocation': False,
            'Location': False,
            'LBI': ':.4f',
            'LII': ':.4f',
            'S_T': ':.2%',
            'MF_ratio': ':.3f'
        },
        color_continuous_scale=color_scales.get(metric, 'YlOrRd'),
        title=f'{metric_titles.get(metric, metric)} by Country ({year})'
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='lightgray',
            projection_type='natural earth',
            showland=True,
            landcolor='#f8f9fa',
            showocean=True,
            oceancolor='#e6f2ff',
            showcountries=True,
            countrycolor='lightgray'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(
            title=metric,
            tickformat='.2f' if metric != 'S_T' else '.0%'
        )
    )

    # Enable scroll zoom for map interaction
    fig.update_geos(
        visible=True,
        resolution=110,
        showcountries=True
    )

    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>Loneliness Risk Index Dashboard</h1>
        <p>Interactive visualization of demographic loneliness risk based on Lokshin & Foster methodology</p>
    </div>
    """, unsafe_allow_html=True)

    # Introduction with data source link
    st.markdown(f"""
    Compare **loneliness burden** across countries and regions using population data from the
    [UN World Population Prospects 2024]({UN_WPP_URL}). The index measures demographic
    imbalances that may contribute to social isolation among elderly populations.
    """)

    st.divider()

    # Load data
    df = load_un_data()

    if df is None:
        st.stop()

    # Get unique locations and years
    locations = sorted(df['Location'].unique())
    years = sorted(df['Time'].unique())

    # Separate countries from regions (heuristic: regions often have specific patterns)
    # For now, just use all locations

    # ========== SIDEBAR CONTROLS ==========
    st.sidebar.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.sidebar.title("Dashboard Controls")

    # About section in expander
    with st.sidebar.expander("About this Dashboard", expanded=False):
        st.markdown(f"""
        This dashboard visualizes the **Loneliness Risk Index (LRI)**
        developed by Lokshin & Foster to measure demographic factors
        contributing to elderly loneliness.

        **Key Indices:**
        - **LII**: Loneliness Intensity Index
        - **LBI**: Loneliness Burden Index

        **Data Source:**
        [UN World Population Prospects 2024]({UN_WPP_URL})
        """)

    st.sidebar.divider()

    # Country/Region selection
    st.sidebar.subheader("Location Selection")

    primary_loc = st.sidebar.selectbox(
        "Primary Location",
        options=locations,
        index=locations.index('Japan') if 'Japan' in locations else 0
    )

    comparator_loc = st.sidebar.selectbox(
        "Comparator Location",
        options=[loc for loc in locations if loc != primary_loc],
        index=([loc for loc in locations if loc != primary_loc].index('Germany')
               if 'Germany' in locations and 'Germany' != primary_loc else 0)
    )

    # Index parameters
    st.sidebar.subheader("Index Parameters")

    alpha = st.sidebar.slider(
        "α (vulnerability parameter)",
        min_value=0.5,
        max_value=2.5,
        value=1.5,
        step=0.1,
        help="Controls how vulnerability increases with age. Paper uses α=1.5"
    )

    T = st.sidebar.slider(
        "T (elderly threshold)",
        min_value=55,
        max_value=70,
        value=60,
        step=5,
        help="Age threshold for elderly population"
    )

    c_max = st.sidebar.slider(
        "c_max (maximum age)",
        min_value=85,
        max_value=100,
        value=98,
        step=1,
        help="Maximum age in analysis"
    )

    # Year selection
    st.sidebar.subheader("Time Selection")

    year = st.sidebar.slider(
        "Cross-section year",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=2023,
        step=1
    )

    year_range = st.sidebar.slider(
        "Time series range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(1950, 2100),
        step=5
    )

    # ========== CALCULATE INDICES ==========
    df_primary_year = df[(df['Location'] == primary_loc) & (df['Time'] == year)]
    df_comparator_year = df[(df['Location'] == comparator_loc) & (df['Time'] == year)]

    primary_data = calculate_loneliness_indices(df_primary_year, T, alpha, c_max)
    comparator_data = calculate_loneliness_indices(df_comparator_year, T, alpha, c_max)

    if primary_data is None or comparator_data is None:
        st.error("Could not calculate indices. Check if data exists for selected locations/year.")
        st.stop()

    # Time series (using fast cached calculation)
    df_hash = hash(tuple(df['Location'].unique()))  # Simple hash for cache key
    primary_ts = get_time_series_cached(df_hash, primary_loc, T, alpha, c_max, df)
    comparator_ts = get_time_series_cached(df_hash, comparator_loc, T, alpha, c_max, df)

    # Filter to selected year range
    primary_ts = primary_ts[(primary_ts['Year'] >= year_range[0]) & (primary_ts['Year'] <= year_range[1])]
    comparator_ts = comparator_ts[(comparator_ts['Year'] >= year_range[0]) & (comparator_ts['Year'] <= year_range[1])]

    # ========== DISPLAY RESULTS ==========

    # Summary metrics
    st.header("📈 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_lii = primary_data['LII'] - comparator_data['LII']
        st.metric(
            label=f"LII - {primary_loc}",
            value=f"{primary_data['LII']:.4f}",
            delta=f"{delta_lii:+.4f}" if delta_lii != 0 else None
        )

    with col2:
        st.metric(
            label=f"LII - {comparator_loc}",
            value=f"{comparator_data['LII']:.4f}"
        )

    with col3:
        delta_lbi = primary_data['LBI'] - comparator_data['LBI']
        st.metric(
            label=f"LBI - {primary_loc}",
            value=f"{primary_data['LBI']:.4f}",
            delta=f"{delta_lbi:+.4f}" if delta_lbi != 0 else None
        )

    with col4:
        st.metric(
            label=f"LBI - {comparator_loc}",
            value=f"{comparator_data['LBI']:.4f}"
        )

    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(label=f"S_T (elderly share) - {primary_loc}", value=f"{primary_data['S_T']:.2%}")

    with col6:
        st.metric(label=f"S_T (elderly share) - {comparator_loc}", value=f"{comparator_data['S_T']:.2%}")

    with col7:
        st.metric(label=f"M/F ratio 60+ - {primary_loc}", value=f"{primary_data['MF_ratio']:.3f}")

    with col8:
        st.metric(label=f"M/F ratio 60+ - {comparator_loc}", value=f"{comparator_data['MF_ratio']:.3f}")

    # ========== TABS FOR VISUALIZATIONS ==========
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️ World Map",
        "📉 Time Series",
        "📊 Age-Specific Curves",
        "🔬 Components",
        "👥 Population Pyramids",
        "📋 Data Table"
    ])

    with tab1:
        st.subheader(f"Global Loneliness Burden Index ({year})")

        # Calculate global LBI for the selected year
        global_lbi = calculate_global_lbi(df_hash, year, T, alpha, c_max, df)

        # Metric selector for map
        map_metric = st.radio(
            "Select metric to display:",
            options=['LBI', 'LII', 'S_T'],
            format_func=lambda x: {'LBI': 'Loneliness Burden Index (LBI)',
                                   'LII': 'Loneliness Intensity Index (LII)',
                                   'S_T': 'Share of Elderly Population'}[x],
            horizontal=True
        )

        # World map
        fig_map = plot_world_map(global_lbi, year, map_metric)
        st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})

        # Top/Bottom countries
        col_top, col_bottom = st.columns(2)

        with col_top:
            st.markdown(f"**Top 10 Countries by {map_metric}**")
            top_10 = global_lbi.nlargest(10, map_metric)[['Location', 'LBI', 'LII', 'S_T']].reset_index(drop=True)
            top_10.index = top_10.index + 1
            st.dataframe(top_10.style.format({'LBI': '{:.4f}', 'LII': '{:.4f}', 'S_T': '{:.2%}'}),
                        use_container_width=True)

        with col_bottom:
            st.markdown(f"**Bottom 10 Countries by {map_metric}**")
            bottom_10 = global_lbi.nsmallest(10, map_metric)[['Location', 'LBI', 'LII', 'S_T']].reset_index(drop=True)
            bottom_10.index = bottom_10.index + 1
            st.dataframe(bottom_10.style.format({'LBI': '{:.4f}', 'LII': '{:.4f}', 'S_T': '{:.2%}'}),
                        use_container_width=True)

    with tab2:
        st.subheader("Loneliness Indices Over Time")

        col1, col2 = st.columns(2)

        with col1:
            fig_lii_ts = plot_time_series_comparison(primary_ts, comparator_ts, 'LII')
            st.plotly_chart(fig_lii_ts, use_container_width=True)

        with col2:
            fig_lbi_ts = plot_time_series_comparison(primary_ts, comparator_ts, 'LBI')
            st.plotly_chart(fig_lbi_ts, use_container_width=True)

        # Elderly share over time
        fig_st_ts = plot_time_series_comparison(primary_ts, comparator_ts, 'S_T')
        st.plotly_chart(fig_st_ts, use_container_width=True)

    with tab3:
        st.subheader(f"Age-Specific Analysis ({year})")

        # LI_c curves
        fig_li = plot_loneliness_intensity_curves(primary_data, comparator_data, primary_loc, comparator_loc, year)
        st.plotly_chart(fig_li, use_container_width=True)

        # Burden contribution curves
        fig_lb = plot_loneliness_burden_curves(primary_data, comparator_data, primary_loc, comparator_loc, year)
        st.plotly_chart(fig_lb, use_container_width=True)

    with tab4:
        st.subheader("Index Components")

        st.markdown(f"""
        **Formula**: LI_c = |g_c| × V_c(α), where:
        - g_c = (F_c - M_c)/(F_c + M_c) — normalized gender gap
        - V_c(α) = ((c - T + 1)/(T + 1))^α — vulnerability factor (α = {alpha})
        - s_c = (M_c + F_c)/N_T — cohort share
        """)

        fig_comp1 = plot_components(primary_data, primary_loc, year, alpha)
        st.plotly_chart(fig_comp1, use_container_width=True)

        fig_comp2 = plot_components(comparator_data, comparator_loc, year, alpha)
        st.plotly_chart(fig_comp2, use_container_width=True)

    with tab5:
        st.subheader("Elderly Population Structure")

        col1, col2 = st.columns(2)

        with col1:
            fig_pyr1 = plot_population_pyramid(primary_data, primary_loc, year)
            st.plotly_chart(fig_pyr1, use_container_width=True)

        with col2:
            fig_pyr2 = plot_population_pyramid(comparator_data, comparator_loc, year)
            st.plotly_chart(fig_pyr2, use_container_width=True)

    with tab6:
        st.subheader("Detailed Data")

        # Create comparison table
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
            col: '{:.4f}' for col in comparison_df.columns if col != 'Age'
        }), use_container_width=True)

    # ========== DOWNLOAD SECTION ==========
    st.header("📥 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        ts_combined = pd.concat([primary_ts, comparator_ts])
        csv_ts = ts_combined.to_csv(index=False)
        st.download_button(
            label="Download Time Series (CSV)",
            data=csv_ts,
            file_name=f"loneliness_ts_{primary_loc}_{comparator_loc}.csv",
            mime="text/csv"
        )

    with col2:
        # Cross-section data
        cross_df = pd.DataFrame({
            'Age': primary_data['ages'],
            f'{primary_loc}_PopMale': primary_data['pop_male'],
            f'{primary_loc}_PopFemale': primary_data['pop_female'],
            f'{primary_loc}_g_c': primary_data['g_c'],
            f'{primary_loc}_V_c': primary_data['V_c'],
            f'{primary_loc}_s_c': primary_data['s_c'],
            f'{primary_loc}_LI_c': primary_data['LI_c'],
            f'{comparator_loc}_PopMale': comparator_data['pop_male'],
            f'{comparator_loc}_PopFemale': comparator_data['pop_female'],
            f'{comparator_loc}_g_c': comparator_data['g_c'],
            f'{comparator_loc}_LI_c': comparator_data['LI_c'],
        })
        csv_cross = cross_df.to_csv(index=False)
        st.download_button(
            label="Download Cross-Section (CSV)",
            data=csv_cross,
            file_name=f"loneliness_cross_{primary_loc}_{comparator_loc}_{year}.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <h4>About</h4>
        <p>
            <strong>Methodology:</strong> Lokshin, M. and J. Foster. "Loneliness Risk Index."
        </p>
        <p>
            <strong>Data Source:</strong>
            <a href="{UN_WPP_URL}" target="_blank">UN World Population Prospects 2024</a>
            (<a href="{UN_WPP_DOWNLOAD_URL}" target="_blank">Download Data</a>)
        </p>
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
            <li><strong>LI<sub>c</sub></strong> = |g<sub>c</sub>| × V<sub>c</sub>(α) × s<sub>c</sub> — Age-specific loneliness index</li>
            <li><strong>LII</strong> = Σ LI<sub>c</sub> — Loneliness Intensity Index</li>
            <li><strong>LBI</strong> = S<sub>T</sub> × LII — Loneliness Burden Index</li>
        </ul>
        <p style="margin-top: 1rem; font-size: 0.85rem; color: #6c757d;">
            Where: g<sub>c</sub> = normalized gender ratio, V<sub>c</sub> = vulnerability factor,
            s<sub>c</sub> = cohort share, S<sub>T</sub> = elderly population share
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
