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
Static hosting (GitHub Pages)
  │
  ├── index.html + JS bundle (~300KB gzipped)
  └── data/
      ├── z-score-peaks.parquet (default dataset)
      ├── v1.parquet, v2.parquet (versioned datasets)
      └── versions.json (auto-generated manifest)

Browser
  ├── Fetch Parquet once → parse with hyparquet → store in memory
  ├── Filter chain (TypeScript) → <50ms for 67K rows
  ├── Plotly.js → boxplots, violins, heatmaps, stats table
  ├── Version comparison across datasets
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
│   │   │   ├── types.ts         # Row interface, FilterParams, VersionInfo
│   │   │   ├── loader.ts        # Parquet fetch + CSV upload parsing
│   │   │   ├── filterChain.ts   # 10-step filter orchestrator
│   │   │   ├── filterSteps.ts   # Individual filter functions
│   │   │   ├── dropdownOptions.ts  # FG conditioning logic
│   │   │   ├── comparison.ts    # Version comparison logic
│   │   │   └── uploadStorage.ts # CSV upload storage utilities
│   │   ├── plots/
│   │   │   ├── boxplot.ts       # Plotly boxplot config
│   │   │   ├── heatmap.ts       # Plotly heatmap config
│   │   │   ├── violin.ts        # Plotly violin plot config
│   │   │   ├── helpers.ts       # Data grouping, median traces, rank annotations
│   │   │   ├── colors.ts        # ELN density color mapping
│   │   │   └── types.ts         # PlotConfig interface
│   │   ├── components/
│   │   │   ├── LandingPage.tsx  # Reaction type search + filter setup
│   │   │   ├── Dashboard.tsx    # Main dashboard layout
│   │   │   ├── Navbar.tsx       # Nav bar + settings + upload
│   │   │   ├── SettingsMenu.tsx # Settings modal
│   │   │   ├── FilterControls.tsx  # 4 multi-select dropdowns
│   │   │   ├── OptionsPanel.tsx # Sliders, checkboxes, downloads
│   │   │   ├── AnalysisTabs.tsx # Tab switcher: boxplot/violin/heatmap/stats
│   │   │   ├── DistributionView.tsx  # Generic renderer for boxplot + violin
│   │   │   ├── HeatmapView.tsx  # Plotly heatmap renderer
│   │   │   ├── StatsTable.tsx   # Descriptive statistics table
│   │   │   ├── MultiSelect.tsx  # Reusable dropdown component
│   │   │   ├── TutorialOverlay.tsx  # Guided tour overlay
│   │   │   ├── Plot.tsx         # Plotly wrapper (dist-min bundle)
│   │   │   └── Footer.tsx       # Paper citation
│   │   ├── hooks/
│   │   │   ├── useFilteredData.ts     # useMemo wrapper for filter chain
│   │   │   ├── useUrlState.ts         # Bidirectional URL ↔ state sync
│   │   │   ├── useTutorial.ts         # Tutorial state machine
│   │   │   ├── useComparisonData.ts   # Version comparison data
│   │   │   ├── useEffectiveDataset.ts # Dataset selection logic
│   │   │   └── useSplitFilteredData.ts # Split data across categories
│   │   └── styles/
│   │       └── app.css          # All styles
│   ├── public/
│   │   ├── data/
│   │   │   ├── z-score-peaks.parquet  # Default dataset (gitignored)
│   │   │   ├── v1.parquet, v2.parquet # Versioned datasets
│   │   │   ├── versions.json          # Auto-generated manifest
│   │   │   └── *-dropdown-index.json  # Precomputed dropdown options
│   │   └── assets/              # Logo, hiker icon
│   ├── golden/                  # Golden test fixtures
│   ├── src/__tests__/           # Vitest tests
│   ├── vite-plugin-versions.ts  # Custom Vite plugin for dataset versioning
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── scripts/
│   └── version_dataset.py       # Dataset versioning
├── .devcontainer/               # Dev container config + lifecycle scripts
├── .github/workflows/           # CI/CD (deploy, pages)
├── pyproject.toml               # Ruff config
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
| Version dataset | `python scripts/version_dataset.py` |
| Lint | `cd frontend && npx eslint .` |

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
**Row interface (17 typed columns + index signature):**
`ELN_ID`, `PLATENUMBER`, `Coordinate`, `AREA_TOTAL_REDUCED`, `Base`, `Catalyst`,
`Solvent`, `Ligand`, `Additive`, `Coupling Reagent`, `Secondary Solvent`,
`Reaction Type`, `FG A`, `FG B`, `FG_sorted`, `FG_PAIR_SORTED` (computed),
`z-Score`

**Key constants in `types.ts`:**
- `DEFAULTS` — single source of truth for filter defaults
- `CATEGORY_OPTIONS` — 7 reactant columns available for grouping
- `REQUIRED_COLUMNS` — 13 columns required in uploaded CSVs
- `REAGENT_COLS` — 8 columns used in deduplication + scale-up detection

## Testing
- **TypeScript tests** (Vitest) — unit tests for filter steps, store, URL state, colors, boxplot config, plus golden fixtures for dropdowns and stats
- Golden fixtures in `frontend/golden/` cover dropdowns, stats, and parity with the former Python implementation

## Key Patterns
- **Two-phase data load:** dropdown index (~12KB, instant) loads first for UI, then Parquet streams in background
- **Split mode:** any filter with 2+ values can be split into side-by-side panels (filter chain runs once per panel)
- **Version comparison:** rank-delta based (median z-Score per group), shown as badge annotations on plots
- **Generic distribution view:** `DistributionView.tsx` renders both boxplot and violin via a `buildConfig` function prop
- **DEFAULTS constant:** `types.ts` is the single source of truth for all filter defaults

## Skills
Always follow the guidelines defined in these skill files:
- `.claude/skills/git/SKILL.md` — Git commit and branching rules
- `.claude/skills/dev/SKILL.md` — Development workflow
- `.claude/skills/build/SKILL.md` — Production build
- `.claude/skills/lint/SKILL.md` — Linting and type checking
- `.claude/skills/tutorial/SKILL.md` — Tutorial system modification guidelines
