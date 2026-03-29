# Z-Score Dashboard — Frontend

Interactive dashboard for analyzing z-score data from high-throughput chemistry experiments (HTE).

## Quick Start

```bash
npm install
npm run dev        # Dev server at http://localhost:5173
npm run build      # Production build
npx tsc --noEmit   # Type check
npx vitest run     # Run tests
```

## Dataset Versioning

The app supports multiple dataset versions. The newest version loads by default, and users can switch between versions via the settings menu.

### How it works

- Versioned datasets live in `public/data/` as `v1.parquet`, `v2.parquet`, etc.
- Each version has a companion dropdown index (`v1-dropdown-index.json`)
- A `versions.json` manifest lists all versions with metadata (label, date, file paths)
- A Vite plugin auto-discovers versioned files at build time and serves the manifest dynamically in dev mode

### Adding a new version

See [`../add-dataset/README.md`](../add-dataset/README.md) for instructions. In short: drop a CSV into `add-dataset/` and commit.

## User Data Upload

Users can upload their own CSV datasets via the settings menu:

- **My data only** — replaces the built-in dataset
- **Combined with built-in** — merges uploaded rows with the active version (ELN IDs prefixed with `upload_` to avoid collisions)
- Uploads are cached in the browser's localStorage and survive page refresh
- Users can switch modes or remove uploaded data at any time

## Architecture

```
Browser
  ├── Fetch Parquet once → parse with hyparquet → store in memory
  ├── Filter chain (10 steps, <50ms for 70K rows)
  ├── Plotly.js → boxplots, violin plots, heatmaps
  ├── URL state sync → shareable deep links
  └── PNG export via Plotly.toImage()
```

## Tech Stack

- React 19 + TypeScript + Vite
- Plotly.js via react-plotly.js
- Zustand (state management)
- React Router v7 (routing + URL state)
- hyparquet (Parquet parsing, pure JS)
- PapaParse (CSV upload parsing)
