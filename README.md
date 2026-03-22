# Z-Score Dashboard

Interactive dashboard for analyzing z-score data from high-throughput chemical reaction screening experiments (HTE).

Built for Roche Pharma R&D (Team RoSL) to explore reaction conditions across 66,000+ experiments and 42 reaction types.

## Quick Start

```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
```

## Features

- **Boxplot analysis** — z-Score distributions by catalyst, solvent, base, ligand, etc., with ELN density coloring
- **Heatmap view** — Median z-Score pivots across two reagent dimensions
- **Statistics table** — Descriptive statistics (count, mean, std, quartiles) per category
- **10-step filter chain** — Reaction types, functional groups, CuI exclusion, scale-up removal, deduplication, top-N, min ELN, max components
- **Client-side filtering** — All 67K rows filtered in <50ms, no server needed
- **Parquet data format** — 0.5MB payload (30x smaller than CSV)
- **URL deep linking** — Every filter state is encoded in the URL
- **CSV upload** — Bring your own dataset with validation
- **PNG/CSV export** — Download filtered data or plot images
- **Interactive tutorial** — 11-step guided walkthrough
- **Presentation mode** — Scaled-up fonts for projectors

## Tech Stack

React 19, TypeScript, Vite, Plotly.js, Zustand, React Router, hyparquet

## Testing

```bash
cd frontend
npx vitest run     # 3,192 golden tests
```

Validates the TypeScript filter chain against Python golden fixtures:
- 2,275 median consistency snapshots (35 filter combos x 42 reaction types)
- 207 dropdown conditioning tests (FG B options per FG A)
- 529 heatmap pivot tests (cell values, axis ordering)
- 181 stats table tests (descriptive statistics)

## Paper

Publication scripts live in [`paper/`](paper/README.md) -- Shapiro-Wilk, Kruskal-Wallis, Mann-Whitney tests, batch figure export. These use scipy and are not part of the deployed dashboard.

> Ahlbrecht, J.; Lutz, M. D. R.; Jost, V.; Farber, M.; Brase, S.; Wuitschik, G.
> *Which Reaction Conditions Work on Drug-Like Molecules? Lessons from 66,000 High-Throughput Experiments.*
> ACS Cent. Sci. **2026**, 12 (2), 222-232.
> [DOI: 10.1021/acscentsci.5c02031](https://doi.org/10.1021/acscentsci.5c02031)

## License

[GPL-3.0](LICENSE)
