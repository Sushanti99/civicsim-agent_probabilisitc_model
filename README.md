# CivicSim — Bayesian demographic agent generator

Generates synthetic agents (age, race, occupation, income) for a location using distributions from census-style data. Default: **Alameda County, 25 agents**, stratified to match population.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Prints the agent table and writes **agents.txt** (same table).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--location` | alameda,california | Location (data in `data/`) |
| `--n_agents` | 25 | Number of agents |
| `--seed` | — | Random seed |
| `--no-diverse` | — | Random sampling instead of stratified |
| `--validate` | — | Print population vs sample report |
| `-o FILE` | — | Also write output to FILE |

## Data

- **Input:** `data/age.csv`, `data/race.csv`, `data/occupation.csv`, `data/income.csv` (from `Demographic Data.xlsx`).
- To refresh CSVs after changing the Excel file, run **data_cleaning.ipynb** (one cell).
