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

3. (Optional) Download UN World Population Prospects data:
   - Visit: https://population.un.org/wpp/Download/Standard/CSV/
   - Download "Population by Age and Sex" data
   - Save as `data/WPP_Population.csv`

## Usage

### Option 1: Double-click
Run `run_dashboard.bat`

### Option 2: Command line
```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Using Real UN Data

To use actual UN World Population Prospects data instead of sample data:

1. Go to https://population.un.org/wpp/Download/Standard/CSV/
2. Download the population by single age CSV file
3. Process it to have columns: `Location`, `Time`, `AgeStart`, `PopMale`, `PopFemale`
4. Save as `data/WPP_Population.csv`

Alternatively, create a script to fetch data from the UN Population API.

## Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| α (alpha) | Vulnerability parameter - controls how vulnerability increases with age | 1.5 | 0.0 - 3.0 |
| T | Elderly threshold age | 60 | 55 - 70 |
| c_max | Maximum age in analysis | 100 | 85 - 100 |

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
- `g_c = (F_c - M_c)/(F_c + M_c)` - normalized gender gap
- `V_c(α) = ((c-T+1)/(T+1))^α` - vulnerability factor
- `s_c` - cohort share in elderly population
- `S_T` - share of elderly in total population

## Reference

Lokshin, M. and J. Foster. "Loneliness Risk Index: Measuring Demographic Risks of Loneliness in Aging Populations."
