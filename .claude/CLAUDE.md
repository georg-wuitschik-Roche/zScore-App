# zScore-App

## Project Overview
- **Goal:** Interactive dashboard for analyzing z-score data from chemical reaction screening experiments (HTE)
- **Users:** Chemists at Roche Pharma R&D (Team RoSL) analyzing reaction conditions
- **Status:** Production — fully client-side React app, no backend needed
- **Environment:** Dev container

## Tech Stack
- **Framework:** React 19 + TypeScript + Vite
- **Charts:** Plotly.js via react-plotly.js
- **State:** Zustand (filter state, UI state)
- **Routing:** React Router v7
- **Data:** Parquet format (0.5MB), parsed with hyparquet
- **Styling:** CSS with custom properties (DM Sans + JetBrains Mono)
- **Testing:** Vitest

## Architecture
```
Static hosting (GitHub Pages / Cloudflare / GCS)
  │
  ├── index.html + JS bundle (~300KB gzipped)
  └── data/z-score-peaks.parquet (0.5MB)

Browser
  ├── Fetch Parquet once → parse with hyparquet → store in memory
  ├── Filter chain (TypeScript) → <50ms for 67K rows
  ├── Plotly.js → boxplots, heatmaps (same as original)
  ├── URL state (React Router search params) → deep linking
  └── PNG export via Plotly.toImage()
```

## Repository Structure
```
zScore-App/
├── frontend/                    # React application
│   ├── src/
│   │   ├── App.tsx              # Router + layout
│   │   ├── main.tsx             # Entry point + error boundary
│   │   ├── stores/
│   │   │   └── filterStore.ts   # Zustand store (all state)
│   │   ├── data/
│   │   │   ├── types.ts         # Row interface, FilterParams
│   │   │   ├── loader.ts        # Parquet fetch + CSV upload parsing
│   │   │   ├── filterChain.ts   # 10-step filter orchestrator
│   │   │   ├── filterSteps.ts   # Individual filter functions
│   │   │   └── dropdownOptions.ts  # FG conditioning logic
│   │   ├── plots/
│   │   │   ├── boxplot.ts       # Plotly boxplot config
│   │   │   ├── heatmap.ts       # Plotly heatmap config
│   │   │   ├── colors.ts        # ELN density color mapping
│   │   │   └── types.ts         # PlotConfig interface
│   │   ├── components/
│   │   │   ├── LandingPage.tsx  # Reaction type search + filter setup
│   │   │   ├── Dashboard.tsx    # Main dashboard layout
│   │   │   ├── Navbar.tsx       # Nav bar + settings + upload
│   │   │   ├── FilterControls.tsx  # 4 multi-select dropdowns
│   │   │   ├── OptionsPanel.tsx # Sliders, checkboxes, downloads
│   │   │   ├── AnalysisTabs.tsx # Boxplot/Heatmap/Stats toggle
│   │   │   ├── BoxplotView.tsx  # Plotly boxplot renderer
│   │   │   ├── HeatmapView.tsx  # Plotly heatmap renderer
│   │   │   ├── StatsTable.tsx   # Descriptive statistics table
│   │   │   ├── MultiSelect.tsx  # Reusable dropdown component
│   │   │   ├── TutorialOverlay.tsx  # 11-step guided tour
│   │   │   ├── Plot.tsx         # Plotly wrapper (dist-min bundle)
│   │   │   └── Footer.tsx       # Paper citation
│   │   ├── hooks/
│   │   │   ├── useFilteredData.ts  # useMemo wrapper for filter chain
│   │   │   ├── useUrlState.ts   # Bidirectional URL ↔ state sync
│   │   │   └── useTutorial.ts   # Tutorial state machine
│   │   └── styles/
│   │       └── app.css          # All styles
│   ├── public/
│   │   ├── data/z-score-peaks.parquet  # Dataset (gitignored, built from CSV)
│   │   └── assets/              # Logo, hiker icon
│   ├── golden/                  # Golden test fixtures (from Python)
│   ├── src/__tests__/           # Vitest tests
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── paper/                       # Publication scripts (not deployed)
│   ├── data_utils.py            # Python filter chain (for stats)
│   ├── plot_utils.py            # Python plot generation
│   ├── stats.py                 # Scipy statistical tests
│   ├── export_boxplots.py       # Batch PNG/SVG export
│   ├── generate_supplementary_figures.py
│   ├── requirements.txt         # Python deps (scipy, pandas, etc.)
│   ├── tests/                   # Python test suite (2,610 tests)
│   └── README.md
├── z-Score Peaks with FG.csv    # Source dataset (~15MB)
├── pyproject.toml               # Ruff/MyPy/pytest config
├── .pre-commit-config.yaml
└── LICENSE
```

## Development

| Task | Command |
|------|---------|
| Install | `cd frontend && npm install` |
| Dev server | `cd frontend && npm run dev` |
| Build | `cd frontend && npm run build` |
| TypeScript check | `cd frontend && npx tsc --noEmit` |
| Tests (all) | `cd frontend && npx vitest run` |
| Tests (watch) | `cd frontend && npx vitest` |
| Regenerate Parquet | `python -c "import pandas as pd; ..."` (see below) |

### Regenerating the Parquet file
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('z-Score Peaks with FG.csv', encoding='utf-8')
USED = ['ELN_ID','PLATENUMBER','Coordinate','AREA_TOTAL_REDUCED',
        'Base','Catalyst','Solvent','Ligand','Additive',
        'Coupling Reagent','Secondary Solvent','Tertiary Solvent',
        'Reaction Type','FG A','FG B','FG_sorted','z-Score','output_column']
df[[c for c in USED if c in df.columns]].to_parquet(
    'frontend/public/data/z-score-peaks.parquet', compression='zstd', index=False)
"
```

## Filter Chain (10 steps)
1. Reaction types → 2. Reactant columns → 3. CuI exclusion → 4. FG A mask →
5. FG B pairs → 6. Scale-up exclusion → 7. Deduplication → 8. Top-N z-scores →
9. Min ELN count → 10. Max components

All filtering runs client-side in TypeScript (<50ms for 67K rows).

## Code Guidelines
- **TypeScript strict mode** — no `any` types, no `as any` casts
- **No backwards compatibility** — delete unused code entirely
- **Zustand for state** — single store, named selectors
- **useMemo for derived data** — filter chain results memoized
- **CSS custom properties** — all design tokens in `:root`
- **Pipe separator** in URLs — avoids comma conflicts in reaction type names

## Data Model
**18 columns in Parquet file:**
`ELN_ID`, `PLATENUMBER`, `Coordinate`, `AREA_TOTAL_REDUCED`, `Base`, `Catalyst`,
`Solvent`, `Ligand`, `Additive`, `Coupling Reagent`, `Secondary Solvent`,
`Tertiary Solvent`, `Reaction Type`, `FG A`, `FG B`, `FG_sorted`, `z-Score`,
`output_column`

## Testing
- **TypeScript tests** (Vitest) — unit tests for filter steps, store, URL state, colors, boxplot config, plus golden fixtures for dropdowns and stats
- **Python tests** (pytest) in `paper/tests/` for the publication scripts
- Golden fixtures in `frontend/golden/` cover dropdowns and stats

## Skills
Always follow the guidelines defined in these skill files:
- `.claude/skills/git.md` — Git commit and branching rules
- `.claude/skills/dev.md` — Development workflow
