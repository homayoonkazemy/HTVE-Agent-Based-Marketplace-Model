# HTVE Agent-Based Marketplace Model

Replication package for the 30-sector agent-based experiments used to study the **Homayoon Theory of Value Exchange (HTVE)** in a closed-loop digital marketplace.

[![Generate replication data](https://github.com/homayoonkazemy/HTVE-Agent-Based-Marketplace-Model/actions/workflows/generate-replication-data.yml/badge.svg)](https://github.com/homayoonkazemy/HTVE-Agent-Based-Marketplace-Model/actions/workflows/generate-replication-data.yml)

## Associated manuscript

**Can a closed-loop digital marketplace scale? Agent-based experiments with the Homayoon Theory of Value Exchange across 30 sectors**

This repository is intended to make the computational results reproducible and auditable. The model is a synthetic agent-based experiment; it is not calibrated to a specific country, firm, or live marketplace.

## Model overview

The baseline model contains:

- 450 heterogeneous participants
- 30 sector archetypes spanning goods and services
- 100 simulated trading periods per run
- endogenous provider acceptance and adaptive internal prices
- positive unit balances and a conserved aggregate unit stock
- a transaction levy split between operating and social-access pools
- optional recycling of pooled units
- explicit capacity constraints
- alternative initial unit distributions

The experimental design contains **460 Monte Carlo runs** across four experiment families:

1. **Market breadth** — varies the fraction of active sectors.
2. **Genesis unit stock** — varies mean initial unit balances.
3. **Genesis concentration** — varies the share of initial units held by the top decile, with and without recycling.
4. **Capacity and recycling** — compares spare versus tight capacity and levy recycling.

## Repository structure

```text
HTVE-Agent-Based-Marketplace-Model/
├── README.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── .github/workflows/
│   └── generate-replication-data.yml
├── code/
│   └── htve_abm_replication.py
├── data/
│   ├── sector_archetypes.csv
│   ├── simulation_all_runs.csv
│   ├── experiment_A_market_breadth_runs.csv
│   ├── experiment_B_unit_stock_runs.csv
│   ├── experiment_C_genesis_concentration_runs.csv
│   ├── experiment_D_capacity_recycling_runs.csv
│   ├── summary_A_market_breadth.csv
│   ├── summary_B_unit_stock.csv
│   ├── summary_C_genesis_concentration.csv
│   └── summary_D_capacity_recycling.csv
└── figures/
    └── generated PNG figures
```

## Reproduction

Python 3 is required. Install dependencies with:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python code/htve_abm_replication.py
```

The script creates a `replication_results/` directory next to the script and reproduces the run-level files, summary tables, sector definitions, and figures.

The repository also includes a GitHub Actions workflow that executes the complete model from a clean Python environment and commits the reproduced data and figures. This provides an independent, machine-executed reproducibility check.

The code uses fixed seed ranges for the four experiment families, so results are reproducible subject to the usual numerical differences that can arise across library/platform versions.

## Data files

- `data/sector_archetypes.csv` — the 30 sector labels used by the model.
- `data/simulation_all_runs.csv` — run-level outcomes for all 460 simulations.
- `data/experiment_A_market_breadth_runs.csv` — all 80 market-breadth runs.
- `data/experiment_B_unit_stock_runs.csv` — all 100 unit-stock runs.
- `data/experiment_C_genesis_concentration_runs.csv` — all 200 concentration/recycling runs.
- `data/experiment_D_capacity_recycling_runs.csv` — all 80 capacity/recycling runs.
- `data/summary_A_market_breadth.csv` — 95% normal-approximation confidence intervals by sector coverage.
- `data/summary_B_unit_stock.csv` — summaries by mean initial unit balance.
- `data/summary_C_genesis_concentration.csv` — summaries by top-decile genesis concentration and recycling condition.
- `data/summary_D_capacity_recycling.csv` — summaries by capacity condition and recycling condition.

## Interpretation boundary

The simulation is designed to test mechanisms, not to forecast a national economy or claim universal thresholds. Numerical tipping points in the model are parameter-dependent. The model does not assume that internal units are exempt from applicable law, taxation, consumer protection, professional regulation, or other institutional requirements.

## Citation

Citation metadata are provided in `CITATION.cff`.

**Author:** Homayoon Kazemy  
**ORCID:** 0000-0003-1929-5999

## License

No open-source or open-data license is granted by this repository at present. Unless a license is added later, normal copyright restrictions apply. The materials are made publicly viewable to support scholarly review and reproducibility.
