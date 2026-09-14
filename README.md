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

## CSV Upload Format

Settings → Data → **Upload Dataset** accepts your own screening data. The ⓘ button next
to it shows this reference in the app and offers a downloadable template CSV.

### Required columns

| Column | Contents |
|---|---|
| `ELN_ID` | Experiment notebook identifier |
| `PLATENUMBER` | Plate number within the experiment |
| `Coordinate` | Well coordinate, e.g. `A1` |
| `AREA_TOTAL_REDUCED` | Total reduced peak area (numeric) |
| `Base` | Base used in the reaction |
| `Catalyst` | Catalyst used in the reaction |
| `Solvent` | Primary solvent |
| `Ligand` | Ligand used in the reaction |
| `Reaction Type` | Reaction class, e.g. `Buchwald-Hartwig amination` |
| `FG A` | Functional group on the first coupling partner |
| `FG B` | Functional group on the second coupling partner |
| `FG_sorted` | Alphabetically sorted `FG A` / `FG B` pair |
| `z-Score` | Normalised performance score (numeric) |

Upload is rejected with a list of the missing names if any of these are absent.

### Optional columns

`Additive`, `Coupling Reagent` and `Secondary Solvent` are used for grouping and
deduplication when present. Leave them empty or omit the columns entirely.
`FG_PAIR_SORTED` is derived from `FG A` and `FG B` at load time — don't supply it.
Any further columns are carried through untouched and ignored by the filter chain.

### Format rules

- Comma, semicolon and tab delimiters are auto-detected (PapaParse).
- `z-Score` must contain numeric values in at least some rows, otherwise the upload
  is rejected. A decimal comma (`1,42`) is normalised to a period.
- Empty strings and `NaN` become `null` for categorical columns.
- Maximum file size is 50 MB.

### Replace vs. combine

After a valid upload you choose how the data is used:

- **My data** (`replace`) — the dashboard shows only your rows.
- **Combined** (`combine`) — your rows are appended to the built-in dataset, with
  `upload_` prefixed onto each `ELN_ID` to avoid collisions.

The upload is kept in `localStorage`, so it survives a page reload until you remove it.

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

> Ahlbrecht, J.; Lutz, M. D. R.; Jost, V.; Farber, M.; Brase, S.; Wuitschik, G.
> *Which Reaction Conditions Work on Drug-Like Molecules? Lessons from 66,000 High-Throughput Experiments.*
> ACS Cent. Sci. **2026**, 12 (2), 222-232.
> [DOI: 10.1021/acscentsci.5c02031](https://doi.org/10.1021/acscentsci.5c02031)

## License

[GPL-3.0](LICENSE)
