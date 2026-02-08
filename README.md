# Loneliness Risk Index Dashboard

Interactive dashboard for calculating and visualizing the Loneliness Risk Index based on Lokshin & Foster's methodology.

## Features

- **Country Comparison**: Select a primary country and compare with another country or region
- **Custom Parameters**: Adjust vulnerability parameter (α), elderly threshold (T), and maximum age
- **Time Series Analysis**: View LII and LBI trends from 1950-2100
- **Age-Specific Curves**: Visualize loneliness burden distribution across age cohorts
- **Component Analysis**: Examine gender gap, vulnerability, and cohort share components
- **Decomposition**: Understand what drives differences between countries
- **Data Export**: Download results as CSV files

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place UN World Population Prospects data in the dashboard directory:
   - The app expects `wpp2024_population.parquet` in the same folder as `app.py`
   - Alternatively, set the `LRI_DATA_PATH` environment variable to a directory containing
     `WPP2024_Population1JanuaryBySingleAgeSex_Medium_1950-2023.csv` and
     `WPP2024_Population1JanuaryBySingleAgeSex_Medium_2024-2100.csv`

## Usage

### Option 1: Double-click
Run `run_dashboard.bat`

### Option 2: Command line
```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| α (alpha) | Vulnerability parameter — controls how vulnerability increases with age | 1.5 | 0.5 – 2.5 |
| T | Elderly threshold age | 60 | 55 – 80 |
| c_max | Maximum age in analysis | 98 | 85 – 100 |

## Formulas

**Loneliness Intensity Index (LII)**:
```
LII = Σ |g_c| × V_c(α) × s_c
```

**Loneliness Burden Index (LBI)**:
```
LBI = S_T × LII
```

Where:
- `g_c = (F_c - M_c)/(F_c + M_c)` — normalized gender gap
- `V_c(α) = ((c - T + 1)/(c_max - T + 1))^α` — vulnerability factor
- `s_c` — cohort share in elderly population
- `S_T` — share of elderly in total population

## Reference

Lokshin, M. and J. Foster. "Loneliness Risk Index: Measuring Demographic Risks of Loneliness in Aging Populations."
