# zScore-App

## Project Overview
- **Goal:** Interactive dashboard for analyzing z-score data from chemical reaction screening experiments (HTE)
- **Users:** Chemists at Roche Pharma R&D (Team RoSL) analyzing reaction conditions
- **Status:** Production — deployed on Google Cloud Run, actively used for research publications
- **Environment:** Dev container — always assume we are working in a development environment

## Tech Stack
- **Framework:** Dash 2.x + Plotly 5.x (Python)
- **Data:** Pandas, NumPy, SciPy (statistical analysis)
- **Export:** Kaleido (plot images), ReportLab (PDF)
- **Server:** Gunicorn (1 worker, 8 threads)
- **Deploy:** Docker (python:3.11-slim) on Google Cloud Run, Heroku fallback
- **Storage:** Local CSV + Google Cloud Storage fallback
- **Python:** 3.11.7

## Architecture
```
Browser
  │
  ▼
┌─────────────────────────────────────────┐
│  app.py  (Dash bootstrap + server)      │
│    ├── layout.py  (UI components)       │
│    ├── callbacks.py  (event handlers)   │
│    │     ├── data_utils.py  (filter/cache/load) ──► CSV / GCS
│    │     └── plot_utils.py  (boxplots/heatmaps)    │
│    └── assets/app.css  (Apple-style UI)            │
├─────────────────────────────────────────┤
│  Batch scripts (not served):            │
│    export_boxplots.py                   │
│    generate_supplementary_figures.py    │
└─────────────────────────────────────────┘
```

## Repository Structure
```
zScore-App/
├── app.py                    # Entry point, Dash app + server
├── layout.py                 # UI layout (dropdowns, tabs, modals)
├── callbacks.py              # All Dash callbacks (~1,360 lines)
├── data_utils.py             # Data loading, filtering, caching (~1,072 lines)
├── plot_utils.py             # Boxplot/heatmap generation (~968 lines)
├── export_boxplots.py        # Batch PNG/SVG export
├── generate_supplementary_figures.py  # Stats figures for papers
├── assets/
│   ├── app.css               # Custom styling
│   └── logo.png              # App logo
├── exports/                  # Generated images (boxplots, paper, supplementary)
├── requirements.txt          # Python deps
├── Dockerfile                # Cloud Run container
├── Procfile                  # Heroku deploy
└── z-Score Peaks with FG.csv # Main dataset (~15 MB)
```

## Development

| Task       | Command                                      |
|------------|-----------------------------------------------|
| Install    | `pip install -r requirements.txt`             |
| Dev server | `python app.py` (debug mode, port 8050)       |
| Production | `gunicorn --bind :8080 -w1 --threads 8 app:server` |
| Export     | `python export_boxplots.py`                   |
| Stats figs | `python generate_supplementary_figures.py`    |

- Local URL: http://localhost:8050
- No test suite exists yet

## Key Files

| Area       | File                 | Purpose                                    |
|------------|----------------------|--------------------------------------------|
| Entry      | `app.py`             | Dash app init, `server` for Gunicorn       |
| UI         | `layout.py`          | All visual components, `serve_layout()`    |
| Logic      | `callbacks.py`       | Event handlers, state management           |
| Data       | `data_utils.py`      | CSV load, filter chain, LRU cache (50)     |
| Viz        | `plot_utils.py`      | `create_boxplot()`, color mapping, hover    |
| Style      | `assets/app.css`     | Apple-inspired design, responsive layout   |
| Deploy     | `Dockerfile`         | Cloud Run container config                 |

## Code Guidelines
- **Separation of concerns:** Layout, callbacks, data, and plot logic live in separate modules. Never mix UI with data processing.
- **Frontend-first filtering:** All filtering happens server-side in `data_utils.filter_data()` via a 10-step chain. Results are cached with MD5-hashed keys.
- **No backwards compatibility:** Delete unused code entirely. No `_var` renames, no `# removed` comments, no re-exports.
- **Type hints everywhere:** Use `from __future__ import annotations`. All functions have docstrings with param/return docs.

## Data Model

**Dataset columns (required):**
`ELN_ID`, `PLATENUMBER`, `Coordinate`, `AREA_TOTAL_REDUCED`, `Base`, `Catalyst`, `Solvent`, `Ligand`, `Reaction Type`, `FG A`, `FG B`, `FG_sorted`, `z-Score`

**9 category types:** Additive, Base, Catalyst, Coupling Reagent, Solvent, FG A, FG B, Ligand, Secondary Solvent

**11 reaction types:** Buchwald-Hartwig, Suzuki-Miyaura, Amide Coupling, Arylation (acidic C-H), Borylation (Miyaura), C-H Activation, C-N Coupling, C-O Coupling, Condensation, Cyclization, Negishi (in-situ)

## Filter Chain (data_utils.filter_data)
1. Reaction types → 2. Reactant columns → 3. CuI exclusion → 4. FG A mask → 5. FG B pairs → 6. Scale-up exclusion → 7. Deduplication → 8. Top-N z-scores → 9. Min ELN count → 10. Max components

## Performance Targets

| Metric                | Target     |
|-----------------------|------------|
| Filter cache entries  | Max 50     |
| Export scale factor   | 4x         |
| Export resolution     | 1600px wide|
| Upload max size       | 50 MB      |
| Gunicorn workers      | 1          |
| Gunicorn threads      | 8          |

## Naming Conventions
- Private functions: `_prefix` (e.g. `_load_and_prepare`)
- Constants: `UPPER_SNAKE` at module top
- Callbacks: underscore prefix (e.g. `_toggle_presentation_mode`)
- Files: `snake_case.py`
