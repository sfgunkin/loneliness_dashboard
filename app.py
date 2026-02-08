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

from iso3_codes import COUNTRY_TO_ISO3

st.set_page_config(
    page_title="Loneliness Risk Index Dashboard",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME & CONSTANTS
# ============================================================================

THEME = {
    'bg': '#dce8dc',
    'card': '#ffffff',
    'border': '#c8d8c8',
    'border_dark': '#b8ccb8',
    'accent': '#3a7d5c',
    'accent_dark': '#2d5a4a',
    'text': '#2d3436',
    'text_muted': '#495057',
}

COLOR = {
    'primary': '#3a7d5c',
    'secondary': '#d4a843',
    'male': '#3498db',
    'female': '#e91e63',
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'purple': '#9b59b6',
}

UN_WPP_URL = "https://population.un.org/wpp/"
UN_WPP_DOWNLOAD_URL = "https://population.un.org/wpp/Download/Standard/CSV/"

# Fields exported in cross-section download
_EXPORT_FIELDS = [('PopMale', 'pop_male'), ('PopFemale', 'pop_female'),
                  ('g_c', 'g_c'), ('V_c', 'V_c'), ('s_c', 's_c'), ('LI_c', 'LI_c')]
_EXPORT_COMP = _EXPORT_FIELDS[:2] + _EXPORT_FIELDS[2:3] + _EXPORT_FIELDS[5:6]  # no V_c, s_c

# Format strings for metric display
_METRIC_FMT = {'LBI': '.3f', 'LII': '.3f', 'S_T': '.2%', 'MF_ratio': '.3f'}
_TABLE_FMT = {k: '{:' + v + '}' for k, v in _METRIC_FMT.items()}


def metric_label(key: str, T: int = 60) -> str:
    return {'LBI': 'Loneliness Burden Index',
            'LII': 'Loneliness Intensity Index',
            'S_T': f'Share of population {T}+',
            'MF_ratio': f'M/F ratio ({T}+)'}[key]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


# ============================================================================
# CSS
# ============================================================================

_pri_fill = hex_to_rgba(COLOR['primary'], 0.10)

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {THEME['bg']}; }}
    [data-testid="stHeader"] {{ display: none; }}
    .block-container {{ padding-top: 0.5rem !important; }}

    .main-header {{
        background: linear-gradient(135deg, {THEME['accent']} 0%, {THEME['accent_dark']} 100%);
        padding: 0.6rem 1.8rem; border-radius: 10px; margin-bottom: 0.6rem;
        color: white; position: sticky; top: 0; z-index: 999;
    }}
    .main-header h1 {{ color: white !important; margin-bottom: 0.1rem; font-size: 1.3rem; }}
    .main-header p {{ color: rgba(255,255,255,0.9); font-size: 0.8rem; margin-bottom: 0; }}

    div[data-testid="stMetric"] {{
        background-color: {THEME['card']}; border: 1px solid {THEME['border_dark']};
        border-radius: 10px; padding: 0.5rem 0.7rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }}
    div[data-testid="stMetric"] label {{
        color: {THEME['text_muted']}; font-weight: 500; font-size: 0.78rem;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 1.5rem; color: {THEME['text']};
    }}

    .main .stMarkdown h2 {{ font-size: 1.2rem; margin-top: 0.3rem; margin-bottom: 0.3rem; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {THEME['card']}; border-radius: 10px 10px 0 0;
        padding: 6px 14px; border: 1px solid {THEME['border']};
        font-size: 0.85rem; color: {THEME['text']};
    }}
    .stTabs [aria-selected="true"] {{ background-color: {THEME['accent']}; color: white; }}
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: {THEME['card']}; border-radius: 0 0 10px 10px;
        padding: 1rem; border: 1px solid {THEME['border']}; border-top: none;
    }}

    section[data-testid="stSidebar"] {{ background-color: {THEME['bg']}; }}
    section[data-testid="stSidebar"] .stMarkdown strong {{ color: {THEME['accent']}; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] {{ font-size: 1.15rem; }}

    .footer {{
        background-color: {THEME['card']}; padding: 1.5rem; border-radius: 10px;
        margin-top: 2rem; border: 1px solid {THEME['border']};
    }}
    .footer a {{ color: {THEME['accent']}; text-decoration: none; }}
    .footer a:hover {{ text-decoration: underline; }}

    .stDownloadButton button {{
        background-color: {THEME['accent']}; color: white;
        border: none; border-radius: 10px; padding: 0.5rem 1rem;
    }}
    .stDownloadButton button:hover {{ background-color: {THEME['accent_dark']}; }}
</style>
"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def color_dot(color: str) -> str:
    return (f'<span style="display:inline-block; width:10px; height:10px; '
            f'border-radius:50%; background:{color}; margin-right:6px; '
            f'vertical-align:middle;"></span>')


def sidebar_label(text: str) -> None:
    st.sidebar.markdown(
        f'<p style="color:{THEME["accent"]}; font-size:1rem; '
        f'font-weight:700; margin:0.8rem 0 0.3rem;">{text}</p>',
        unsafe_allow_html=True)


def filter_elderly(df: pd.DataFrame, T: int, c_max: int) -> pd.DataFrame:
    return df[(df['AgeGrpStart'] >= T) & (df['AgeGrpStart'] <= c_max)].copy()


def vulnerability_factor(ages: np.ndarray, T: int, alpha: float) -> np.ndarray:
    return np.power((ages - T + 1) / (T + 1), alpha)


def safe_divide(num: np.ndarray, den: np.ndarray, fill: float = 1.0) -> np.ndarray:
    return num / np.where(den == 0, fill, den)


# ============================================================================
# PLOTLY LAYOUT HELPERS
# ============================================================================

_FONT_AXIS = 14
_FONT_TICK = 12


def _legend(position: str = 'top-right') -> dict:
    pos = {'top-right': dict(yanchor='top', y=0.99, xanchor='right', x=0.99),
           'top-left': dict(yanchor='top', y=0.99, xanchor='left', x=0.01)}
    return dict(**pos.get(position, pos['top-right']),
                bgcolor='rgba(255,255,255,0.8)', bordercolor='gray', borderwidth=1)


def _xaxis(ages: np.ndarray, title: str = 'Age cohort') -> dict:
    return dict(title=dict(text=title, font=dict(size=_FONT_AXIS)),
                tickfont=dict(size=_FONT_TICK),
                range=[ages.min() - 1, ages.max() + 1],
                dtick=5, showgrid=True, gridcolor='lightgray')


def _yaxis(title: str) -> dict:
    return dict(title=dict(text=title, font=dict(size=_FONT_AXIS)),
                tickfont=dict(size=_FONT_TICK),
                showgrid=True, gridcolor='lightgray',
                hoverformat='.3f')


def _add_pair(fig, primary, comparator, ploc, cloc, xk, yk,
              scale=1.0, fill_primary=False):
    kw = dict(x=primary[xk], y=primary[yk] * scale, mode='lines', name=ploc,
              line=dict(color=COLOR['primary'], width=2.5))
    if fill_primary:
        kw.update(fill='tozeroy', fillcolor=_pri_fill)
    fig.add_trace(go.Scatter(**kw))
    fig.add_trace(go.Scatter(
        x=comparator[xk], y=comparator[yk] * scale, mode='lines', name=cloc,
        line=dict(color=COLOR['secondary'], width=2.5)))


# ============================================================================
# LONELINESS INDEX CALCULATIONS
# ============================================================================

def calculate_indices(df_cy: pd.DataFrame, T: int, alpha: float,
                      c_max: int) -> Optional[Dict[str, Any]]:
    """LI_c = |g_c| * V_c * s_c;  LII = sum(LI_c)*100;  LBI = S_T * LII"""
    eld = filter_elderly(df_cy, T, c_max)
    if eld.empty:
        return None
    eld = eld.sort_values('AgeGrpStart')
    ages = eld['AgeGrpStart'].values
    pm, pf = eld['PopMale'].values, eld['PopFemale'].values

    N_T = np.sum(pm + pf)
    if N_T == 0:
        return None

    total_pop = df_cy['PopTotal'].sum()
    S_T = N_T / total_pop if total_pop > 0 else 0
    s_c = (pm + pf) / N_T
    g_c = safe_divide(pf - pm, pf + pm)
    V_c = vulnerability_factor(ages, T, alpha)
    LI_c = np.abs(g_c) * V_c * s_c
    LII = np.sum(LI_c) * 100

    return dict(ages=ages, pop_male=pm, pop_female=pf,
                s_c=s_c, g_c=g_c, V_c=V_c, LI_c=LI_c,
                LII=LII, LBI=S_T * LII, S_T=S_T, N_T=N_T,
                MF_ratio=np.sum(pm) / np.sum(pf) if np.sum(pf) > 0 else 0)


def decompose_lbi_difference(pd_: dict, cd_: dict) -> dict:
    """Oaxaca-Kitagawa decomposition of ΔLBI (equations 4-6 in the paper).

    Three components:
      Population aging  = ΔS_T × L̄II
      Gender gap        = S̄_T × Σ_c [Δ|g_c| × V_c × s̄_c] × 100
      Age concentration = S̄_T × Σ_c [|ḡ_c| × V_c × Δs_c] × 100
    """
    S_T_A, S_T_B = pd_['S_T'], cd_['S_T']
    LII_A, LII_B = pd_['LII'], cd_['LII']
    delta_LBI = pd_['LBI'] - cd_['LBI']

    S_T_bar = (S_T_A + S_T_B) / 2
    LII_bar = (LII_A + LII_B) / 2

    # Eq (4): first term — population aging effect
    population_aging = (S_T_A - S_T_B) * LII_bar

    # Align age arrays
    common_ages = np.intersect1d(pd_['ages'], cd_['ages'])
    idx_A = {a: i for i, a in enumerate(pd_['ages'])}
    idx_B = {a: i for i, a in enumerate(cd_['ages'])}

    g_A = np.array([np.abs(pd_['g_c'][idx_A[a]]) for a in common_ages])
    g_B = np.array([np.abs(cd_['g_c'][idx_B[a]]) for a in common_ages])
    V_c = np.array([pd_['V_c'][idx_A[a]] for a in common_ages])
    s_A = np.array([pd_['s_c'][idx_A[a]] for a in common_ages])
    s_B = np.array([cd_['s_c'][idx_B[a]] for a in common_ages])

    delta_g = g_A - g_B
    delta_s = s_A - s_B
    g_bar = (g_A + g_B) / 2
    s_bar = (s_A + s_B) / 2

    # Eq (5): age-specific contributions to ΔLII
    gender_gap_by_age = delta_g * V_c * s_bar * 100
    age_conc_by_age = g_bar * V_c * delta_s * 100

    # Eq (6): three-component decomposition of ΔLBI
    gender_gap = S_T_bar * np.sum(gender_gap_by_age)
    age_concentration = S_T_bar * np.sum(age_conc_by_age)

    return dict(delta_LBI=delta_LBI,
                population_aging=population_aging,
                gender_gap=gender_gap,
                age_concentration=age_concentration,
                ages=common_ages,
                gender_gap_by_age=gender_gap_by_age,
                age_conc_by_age=age_conc_by_age)


def _time_series_vectorized(df_loc: pd.DataFrame, location: str,
                             T: int, alpha: float, c_max: int) -> pd.DataFrame:
    eld = filter_elderly(df_loc, T, c_max)
    if eld.empty:
        return pd.DataFrame()

    ages = np.arange(T, c_max + 1)
    eld['V_c'] = eld['AgeGrpStart'].map(dict(zip(ages, vulnerability_factor(ages, T, alpha))))

    pop_total = (eld['PopMale'] + eld['PopFemale']).replace(0, 1)
    eld['g_c'] = (eld['PopFemale'] - eld['PopMale']) / pop_total
    eld['pop_cohort'] = eld['PopMale'] + eld['PopFemale']
    eld['s_c'] = eld['pop_cohort'] / eld.groupby('Time')['pop_cohort'].transform('sum')
    eld['LI_c'] = np.abs(eld['g_c']) * eld['V_c'] * eld['s_c']

    yr = eld.groupby('Time').agg(
        PopMale=('PopMale', 'sum'), PopFemale=('PopFemale', 'sum'),
        LI_c=('LI_c', 'sum')).reset_index()
    yr['N_T'] = yr['PopMale'] + yr['PopFemale']
    yr['MF_ratio'] = yr['PopMale'] / yr['PopFemale'].replace(0, 1)

    tot = df_loc.groupby('Time')['PopTotal'].sum().reset_index()
    tot.columns = ['Time', 'TotalPop']
    yr = yr.merge(tot, on='Time')

    yr['S_T'] = yr['N_T'] / yr['TotalPop']
    yr['LII'] = yr['LI_c'] * 100
    yr['LBI'] = yr['S_T'] * yr['LI_c'] * 100
    yr['Location'] = location
    return yr.rename(columns={'Time': 'Year'})[['Year', 'Location', 'LII', 'LBI', 'S_T', 'MF_ratio']]


@st.cache_data(show_spinner="Calculating global LBI data...")
def _global_lbi(df_hash: int, year: int, T: int, alpha: float,
                c_max: int, _df: pd.DataFrame) -> pd.DataFrame:
    df_year = _df[_df['Time'] == year]
    rows = []
    for loc in df_year['Location'].unique():
        idx = calculate_indices(df_year[df_year['Location'] == loc], T, alpha, c_max)
        if idx:
            rows.append(dict(Location=loc, LII=idx['LII'], LBI=idx['LBI'],
                             S_T=idx['S_T'], MF_ratio=idx['MF_ratio'], N_T=idx['N_T']))
    return pd.DataFrame(rows)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(show_spinner="Loading UN population data...")
def load_un_data() -> Optional[pd.DataFrame]:
    parquet = os.path.join(os.path.dirname(__file__), "wpp2024_population.parquet")
    if os.path.exists(parquet):
        df = pd.read_parquet(parquet)
    else:
        data_path = r"F:\OneDrive\__Documents\Papers\Loneliness Index\Data"
        files = [os.path.join(data_path, f"WPP2024_Population1JanuaryBySingleAgeSex_Medium_{r}.csv")
                 for r in ("1950-2023", "2024-2100")]
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
def _cached_ts(df_hash: int, location: str, T: int, alpha: float,
               c_max: int, _df: pd.DataFrame) -> pd.DataFrame:
    return _time_series_vectorized(_df[_df['Location'] == location], location, T, alpha, c_max)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def _plot_age_curve(pdata, cdata, ploc, cloc, year, burden=False):
    """Age-specific LI(c) or LB(c) = S_T * LI(c) curves."""
    fig = go.Figure()
    if burden:
        pd_ = dict(ages=pdata['ages'], y=pdata['S_T'] * pdata['LI_c'])
        cd_ = dict(ages=cdata['ages'], y=cdata['S_T'] * cdata['LI_c'])
        title = f'Age-Specific Loneliness Burden LB(c) = S_T \u00d7 LI(c), {year}'
        ylabel = 'LB(c) = S_T \u00d7 |g(c)| \u00d7 V(c) \u00d7 s(c)'
    else:
        pd_ = dict(ages=pdata['ages'], y=pdata['LI_c'])
        cd_ = dict(ages=cdata['ages'], y=cdata['LI_c'])
        title = f'Age-Specific Loneliness Index LI(c), {year}'
        ylabel = 'LI(c) = |g(c)| \u00d7 V(c) \u00d7 s(c)'
    _add_pair(fig, pd_, cd_, ploc, cloc, 'ages', 'y', scale=100.0, fill_primary=burden)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=_xaxis(pdata['ages']), yaxis=_yaxis(ylabel),
        legend=_legend(), plot_bgcolor='white', hovermode='x unified', height=500)
    return fig


def plot_time_series(pts, cts, metric='LBI', T=60, show_legend=True):
    fig = go.Figure()
    ploc = pts['Location'].iloc[0] if not pts.empty else 'Location 1'
    cloc = cts['Location'].iloc[0] if not cts.empty else 'Location 2'
    for ts, loc, clr in [(pts, ploc, COLOR['primary']), (cts, cloc, COLOR['secondary'])]:
        fig.add_trace(go.Scatter(
            x=ts['Year'], y=ts[metric], mode='lines+markers', name=loc,
            line=dict(color=clr, width=2), marker=dict(size=4),
            showlegend=show_legend))
    fig.add_vline(x=2024, line_dash="dash", line_color="gray",
                  annotation_text="2024", annotation_position="top right")
    layout = dict(title=metric_label(metric, T),
                  xaxis_title='Year', yaxis=_yaxis(metric_label(metric, T)),
                  hovermode='x unified')
    if show_legend:
        layout['legend'] = dict(orientation='h', yanchor='top', y=-0.18,
                                xanchor='center', x=0.5,
                                bgcolor='rgba(255,255,255,0.8)',
                                bordercolor='gray', borderwidth=1)
    fig.update_layout(**layout)
    return fig


def plot_components(data, location, year, alpha, color=COLOR['primary']):
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        'Gender Ratio g_c', f'Vulnerability V_c(\u03b1={alpha})', 'Cohort Share s_c'))
    fig.add_trace(go.Bar(x=data['ages'], y=data['g_c'], name='g_c',
        marker_color=color),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=data['ages'], y=data['V_c'], mode='lines+markers',
        name='V_c(\u03b1)', line=dict(color=color, width=2)),
        row=1, col=2)
    fig.add_trace(go.Bar(x=data['ages'], y=data['s_c'], name='s_c',
        marker_color=color), row=1, col=3)
    fig.update_layout(title=f'Components of Loneliness Index: {location} ({year})',
                      showlegend=False, height=350)
    ylabels = ['Gender Ratio g(c)', 'Vulnerability V(c)', 'Cohort Share s(c)']
    for c in range(1, 4):
        fig.update_xaxes(title_text='Age', row=1, col=c)
        fig.update_yaxes(title_text=ylabels[c - 1], hoverformat='.3f', row=1, col=c)
    return fig


def plot_pyramid(data, location, year):
    fig = go.Figure()
    mx = max(np.max(data['pop_male']), np.max(data['pop_female']))
    fig.add_trace(go.Bar(y=data['ages'], x=-data['pop_male'], orientation='h',
        name='Males', marker_color=COLOR['male'],
        hovertemplate='Age %{y}<br>Males: %{customdata:.1f}k<extra></extra>',
        customdata=data['pop_male']))
    fig.add_trace(go.Bar(y=data['ages'], x=data['pop_female'], orientation='h',
        name='Females', marker_color=COLOR['female'],
        hovertemplate='Age %{y}<br>Females: %{x:.1f}k<extra></extra>'))
    fig.update_layout(
        title=f'Elderly Population Pyramid: {location} ({year})',
        xaxis=dict(title='Population (thousands)', range=[-mx * 1.1, mx * 1.1],
            tickvals=[-mx, -mx/2, 0, mx/2, mx],
            ticktext=[f'{mx:.0f}', f'{mx/2:.0f}', '0', f'{mx/2:.0f}', f'{mx:.0f}']),
        yaxis_title='Age', barmode='overlay', bargap=0.1, height=450)
    return fig


def plot_world_map(global_lbi, year, metric='LBI', T=60):
    scales = {'LBI': 'YlOrRd', 'LII': 'YlOrRd', 'S_T': 'Blues', 'MF_ratio': 'RdBu'}
    tick_fmts = {'S_T': '.0%', 'MF_ratio': '.2f'}
    df = global_lbi.copy()
    df['ISO3'] = df['Location'].map(COUNTRY_TO_ISO3)
    df = df.dropna(subset=['ISO3'])

    fig = px.choropleth(
        df, locations='ISO3', locationmode='ISO-3', color=metric,
        hover_name='Location',
        hover_data={'ISO3': False, 'Location': False,
                    'LBI': ':.3f', 'LII': ':.3f', 'S_T': ':.2%', 'MF_ratio': ':.3f'},
        color_continuous_scale=scales.get(metric, 'YlOrRd'),
        title=f'{metric_label(metric, T)} by Country ({year})')
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor='lightgray',
                 projection_type='natural earth', showland=True, landcolor='#f8f9fa',
                 showocean=True, oceancolor='#e6f2ff', showcountries=True,
                 countrycolor='lightgray', resolution=110),
        height=600, margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title=metric,
                                tickformat=tick_fmts.get(metric, '.2f')))
    return fig


# ============================================================================
# UI SECTIONS
# ============================================================================

def render_sidebar(locations, years):
    sidebar_label("Countries")
    ploc = st.sidebar.selectbox(
        "Country A", locations,
        index=locations.index('Japan') if 'Japan' in locations else 0)
    comp_opts = [l for l in locations if l != ploc]
    cloc = st.sidebar.selectbox(
        "Country B", comp_opts,
        index=comp_opts.index('Germany') if 'Germany' in comp_opts else 0)

    with st.sidebar.expander("Advanced Parameters", expanded=False):
        alpha = st.slider("\u03b1 (vulnerability)", 0.5, 2.5, 1.5, 0.1,
                          help="Controls how vulnerability increases with age. Paper uses \u03b1=1.5")
        T = st.slider("T (elderly threshold)", 55, 80, 60, 1,
                       help="Age threshold for elderly population")
        c_max = st.slider("c_max (max age)", 85, 100, 98, 1,
                           help="Maximum age in analysis")

    sidebar_label("Time")
    year = st.sidebar.slider("Cross-section year", int(min(years)), int(max(years)), 2023, 1)
    yr_range = st.sidebar.slider("Time series range", int(min(years)), int(max(years)), (1950, 2100), 5)

    return ploc, cloc, alpha, T, c_max, year, yr_range


def render_metrics(pd_, cd_, ploc, cloc, year, T):
    metrics = [
        ("LII", 'LII', '.3f'), ("LBI", 'LBI', '.3f'),
        (f"Elderly ({T}+)", 'S_T', '.2%'), (f"M/F ({T}+)", 'MF_ratio', '.3f'),
    ]
    header = ''.join(f'<th style="padding:0.35rem 1rem; font-weight:600; '
                     f'color:{THEME["text_muted"]}; font-size:0.85rem;">{m[0]}</th>'
                     for m in metrics)

    def _row(label, color, vals, fmts, bold=False, color_vals=False):
        dot = color_dot(color) if color else ''
        w = 'font-weight:700;' if bold else ''
        name = (f'<td style="padding:0.35rem 1rem; white-space:nowrap; {w}">'
                f'{dot}{label}</td>')
        cells = []
        for v, f in zip(vals, fmts):
            vc = ''
            if color_vals:
                vc = f'color:#2196F3;' if v > 0 else (f'color:#FF9800;' if v < 0 else '')
            cells.append(
                f'<td style="padding:0.35rem 1rem; text-align:right; {w}{vc}">'
                f'{v:{f}}</td>')
        return f'<tr>{name}{"".join(cells)}</tr>'

    fmts = [m[2] for m in metrics]
    dfmts = ['+.3f' if f == '.3f' else '+.2%' for f in fmts]
    pvals = [pd_[m[1]] for m in metrics]
    cvals = [cd_[m[1]] for m in metrics]
    dvals = [p - c for p, c in zip(pvals, cvals)]

    table = f"""
    <table style="width:100%; border-collapse:collapse; background:{THEME['card']};
                  border:1px solid {THEME['border_dark']}; border-radius:10px;
                  box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-bottom:0.5rem;">
      <thead><tr style="border-bottom:2px solid {THEME['border']};">
        <th style="padding:0.35rem 1rem; text-align:left; color:{THEME['text_muted']};
                   font-size:0.85rem;">Country ({year})</th>{header}
      </tr></thead>
      <tbody>
        {_row(ploc, COLOR['primary'], pvals, fmts)}
        <tr style="border-bottom:1px solid {THEME['border']};">
          {_row(cloc, COLOR['secondary'], cvals, fmts)[4:-5]}
        </tr>
        {_row('\u0394 Difference', None, dvals, dfmts, bold=True, color_vals=True)}
      </tbody>
    </table>"""
    st.markdown(table, unsafe_allow_html=True)


def render_tab_time_series(pts, cts, T):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(plot_time_series(pts, cts, 'LII', T, show_legend=True), use_container_width=True)
    with c2:
        st.plotly_chart(plot_time_series(pts, cts, 'LBI', T, show_legend=False), use_container_width=True)
    with c3:
        st.plotly_chart(plot_time_series(pts, cts, 'S_T', T, show_legend=False), use_container_width=True)


def render_tab_curves(pd_, cd_, ploc, cloc, year):
    st.plotly_chart(_plot_age_curve(pd_, cd_, ploc, cloc, year), use_container_width=True)
    st.plotly_chart(_plot_age_curve(pd_, cd_, ploc, cloc, year, burden=True), use_container_width=True)


def render_tab_components(pd_, cd_, ploc, cloc, year, alpha):
    st.markdown(f"**Formula**: LI_c = |g_c| \u00d7 V_c(\u03b1) \u00d7 s_c, where: "
                f"g_c = normalized gender gap, V_c(\u03b1) = vulnerability (\u03b1={alpha}), "
                f"s_c = cohort share")
    st.plotly_chart(plot_components(pd_, ploc, year, alpha, COLOR['primary']), use_container_width=True)
    st.plotly_chart(plot_components(cd_, cloc, year, alpha, COLOR['secondary']), use_container_width=True)


def render_tab_decomposition(pd_, cd_, ploc, cloc, year, T):
    decomp = decompose_lbi_difference(pd_, cd_)

    st.markdown(
        f"**\u0394LBI** = LBI({ploc}) \u2212 LBI({cloc}) = "
        f"{pd_['LBI']:.4f} \u2212 {cd_['LBI']:.4f} = **{decomp['delta_LBI']:.4f}**")

    # --- Summary table (paper Table 1 style) -------------------------------
    denom = decomp['delta_LBI'] if abs(decomp['delta_LBI']) > 1e-10 else np.nan
    components = [
        ('Population Aging', decomp['population_aging'],
         'Elderly share difference'),
        ('Gender Gap', decomp['gender_gap'],
         'Male/female longevity difference by age'),
        ('Age Concentration', decomp['age_concentration'],
         'Where within the elderly the gender gaps occur'),
    ]

    cols = st.columns(3)
    for col, (label, value, desc) in zip(cols, components):
        pct = value / denom * 100 if np.isfinite(denom) else np.nan
        vc = '#2196F3' if value > 0 else ('#FF9800' if value < 0 else THEME['text'])
        pct_str = f"{pct:+.1f}% of \u0394LBI" if np.isfinite(pct) else ""
        with col:
            st.markdown(
                f'<div style="background:{THEME["card"]}; border:1px solid {THEME["border_dark"]}; '
                f'border-radius:10px; padding:0.5rem 0.7rem; box-shadow:0 3px 10px rgba(0,0,0,0.1);">'
                f'<div style="color:{THEME["text_muted"]}; font-weight:500; font-size:0.78rem;">{label}</div>'
                f'<div style="font-size:1.5rem; color:{vc}; font-weight:700;">{value:+.4f}</div>'
                f'<div style="font-size:0.85rem; color:{vc};">{pct_str}</div>'
                f'<div style="font-size:0.75rem; color:{THEME["text_muted"]}; margin-top:0.2rem;">{desc}</div>'
                f'</div>', unsafe_allow_html=True)

    # --- Waterfall chart ---------------------------------------------------
    labels = ['Population<br>Aging', 'Gender<br>Gap', 'Age<br>Concentration',
              'Total \u0394LBI']
    values = [decomp['population_aging'], decomp['gender_gap'],
              decomp['age_concentration']]

    fig1 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=labels, y=values + [0],
        connector=dict(line=dict(color="rgb(63,63,63)")),
        increasing=dict(marker=dict(color=COLOR['primary'])),
        decreasing=dict(marker=dict(color=COLOR['secondary'])),
        totals=dict(marker=dict(color=COLOR['purple'])),
        text=[f"{v:+.4f}" for v in values] + [f"{decomp['delta_LBI']:+.4f}"],
        textposition="outside", cliponaxis=False))
    fig1.update_layout(
        title=f"Oaxaca\u2013Kitagawa Decomposition of \u0394LBI ({ploc} \u2212 {cloc})",
        yaxis=_yaxis("Contribution to \u0394LBI"),
        showlegend=False, height=420, plot_bgcolor='white',
        margin=dict(t=60, b=40))
    st.plotly_chart(fig1, use_container_width=True)

    # --- Age-specific decomposition of ΔLII --------------------------------
    st.markdown("#### Age-Specific Decomposition of \u0394LII")
    st.markdown(
        "*The difference in LII decomposes at each age into a gender-gap "
        "effect and an age-concentration effect (eq. 5).*")

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=decomp['ages'], y=decomp['gender_gap_by_age'],
        name='Gender Gap Effect', marker_color=COLOR['female'],
        hovertemplate='Age %{x}<br>Gender gap: %{y:.4f}<extra></extra>'))
    fig2.add_trace(go.Bar(
        x=decomp['ages'], y=decomp['age_conc_by_age'],
        name='Age Concentration Effect', marker_color=COLOR['male'],
        hovertemplate='Age %{x}<br>Age concentration: %{y:.4f}<extra></extra>'))
    fig2.update_layout(
        title=f"Age-Specific Contributions to \u0394LII ({ploc} \u2212 {cloc})",
        xaxis=_xaxis(decomp['ages']),
        yaxis=_yaxis("Contribution to \u0394LII"),
        barmode='relative', height=420, plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='top', y=-0.18,
                    xanchor='center', x=0.5))
    st.plotly_chart(fig2, use_container_width=True)


def render_tab_pyramids(pd_, cd_, ploc, cloc, year):
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_pyramid(pd_, ploc, year), use_container_width=True)
    with c2:
        st.plotly_chart(plot_pyramid(cd_, cloc, year), use_container_width=True)


def render_tab_map(df_hash, year, T, alpha, c_max, df):
    glbi = _global_lbi(df_hash, year, T, alpha, c_max, df)
    metric = st.radio("Select metric:", ['LBI', 'LII', 'S_T', 'MF_ratio'],
        format_func=lambda x: metric_label(x, T), horizontal=True)
    st.plotly_chart(plot_world_map(glbi, year, metric, T),
                    use_container_width=True, config={'scrollZoom': True})

    countries = glbi[glbi['Location'].map(COUNTRY_TO_ISO3).notna()]
    c1, c2 = st.columns(2)
    for col, label, fn in [(c1, "Top 10", 'nlargest'), (c2, "Bottom 10", 'nsmallest')]:
        with col:
            st.markdown(f"**{label} Countries by {metric_label(metric, T)}**")
            tbl = getattr(countries, fn)(10, metric)[['Location', 'LBI', 'LII', 'S_T', 'MF_ratio']].reset_index(drop=True)
            tbl.index = tbl.index + 1
            tbl = tbl.copy()
            tbl['S_T'] = tbl['S_T'] * 100
            st.dataframe(tbl, column_config={
                'LBI': st.column_config.NumberColumn(format='%.3f'),
                'LII': st.column_config.NumberColumn(format='%.3f'),
                'S_T': st.column_config.NumberColumn(format='%.1f%%'),
                'MF_ratio': st.column_config.NumberColumn(format='%.3f'),
            }, use_container_width=True)


def render_footer():
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <p><strong>Methodology:</strong> Lokshin, M. and J. Foster. "Loneliness Risk Index."</p>
        <p><strong>Data:</strong>
            <a href="{UN_WPP_URL}" target="_blank">UN World Population Prospects 2024</a>
            (<a href="{UN_WPP_DOWNLOAD_URL}" target="_blank">Download</a>)</p>
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
        <p>Based on Lokshin & Foster methodology</p>
    </div>
    """, unsafe_allow_html=True)
    df = load_un_data()
    if df is None:
        st.stop()

    locations = sorted(df['Location'].unique())
    years = sorted(df['Time'].unique())
    ploc, cloc, alpha, T, c_max, year, yr_range = render_sidebar(locations, years)

    pd_ = calculate_indices(df[(df['Location'] == ploc) & (df['Time'] == year)], T, alpha, c_max)
    cd_ = calculate_indices(df[(df['Location'] == cloc) & (df['Time'] == year)], T, alpha, c_max)
    if pd_ is None or cd_ is None:
        st.error("Could not calculate indices. Check if data exists for selected locations/year.")
        st.stop()

    df_hash = hash(tuple(df['Location'].unique()))
    pts = _cached_ts(df_hash, ploc, T, alpha, c_max, df)
    cts = _cached_ts(df_hash, cloc, T, alpha, c_max, df)
    pts = pts[(pts['Year'] >= yr_range[0]) & (pts['Year'] <= yr_range[1])]
    cts = cts[(cts['Year'] >= yr_range[0]) & (cts['Year'] <= yr_range[1])]

    render_metrics(pd_, cd_, ploc, cloc, year, T)

    # --- Export in sidebar ---------------------------------------------------
    with st.sidebar.expander("Export data", expanded=False):
        cross = pd.DataFrame({
            'Age': pd_['ages'],
            **{f'{ploc}_{k}': pd_[v] for k, v in _EXPORT_FIELDS},
            **{f'{cloc}_{k}': cd_[v] for k, v in _EXPORT_COMP},
        })
        st.download_button("Cross-section CSV",
            cross.to_csv(index=False),
            f"loneliness_cross_{ploc}_{cloc}_{year}.csv", "text/csv",
            use_container_width=True)
        st.download_button("Time series CSV",
            pd.concat([pts, cts]).to_csv(index=False),
            f"loneliness_ts_{ploc}_{cloc}.csv", "text/csv",
            use_container_width=True)

    # --- Tabs --------------------------------------------------------------
    tab_cur, tab_ts, tab_dec, tab_comp, tab_pyr, tab_map = st.tabs([
        "Age Curves", "Time Series", "Decomposition",
        "Components", "Pyramids", "World Map"])

    with tab_cur:
        render_tab_curves(pd_, cd_, ploc, cloc, year)
    with tab_ts:
        render_tab_time_series(pts, cts, T)
    with tab_dec:
        render_tab_decomposition(pd_, cd_, ploc, cloc, year, T)
    with tab_comp:
        render_tab_components(pd_, cd_, ploc, cloc, year, alpha)
    with tab_pyr:
        render_tab_pyramids(pd_, cd_, ploc, cloc, year)
    with tab_map:
        render_tab_map(df_hash, year, T, alpha, c_max, df)

    render_footer()


if __name__ == "__main__":
    main()
