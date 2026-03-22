# Paper Figures & Statistics

Scripts for generating publication figures and statistical analyses for:

> Ahlbrecht, J.; Lutz, M. D. R.; Jost, V.; Färber, M.; Bräse, S.; Wuitschik, G.
> *Which Reaction Conditions Work on Drug-Like Molecules? Lessons from 66,000 High-Throughput Experiments.*
> ACS Cent. Sci. **2026**, 12 (2), 222–232.

These scripts are **not part of the live dashboard** — they depend on scipy for
statistical tests (Shapiro-Wilk, Kruskal-Wallis, Mann-Whitney, permutation tests)
and Kaleido for publication-quality vector export.

## Setup

```bash
pip install -r paper/requirements.txt
```

## Scripts

| Script | Output | Purpose |
|--------|--------|---------|
| `export_boxplots.py` | `exports/boxplots/` | Batch PNG/SVG boxplots for all reaction types |
| `generate_supplementary_figures.py` | `exports/supplementary/` | Histograms, distribution stats CSV |
| `stats.py` | stdout / importable | Shapiro-Wilk, Kruskal-Wallis, Mann-Whitney, permutation tests |

## Usage

```bash
# From the repo root:
python paper/export_boxplots.py
python paper/generate_supplementary_figures.py
python paper/stats.py
```

## Output

Generated files are saved to `exports/` (gitignored):

```
exports/
├── boxplots/           # Per-reaction-type boxplot PNG/SVG
├── paper_boxplots/     # Publication-quality boxplots
├── supplementary/
│   ├── histograms/     # Distribution histograms
│   └── distribution_stats/  # CSV summary statistics
└── flat_export/        # Flat CSV exports
```

## Relationship to the React Dashboard

The live dashboard (`frontend/`) handles all filtering and visualization
client-side in TypeScript. These paper scripts use the **Python** `data_utils.py`
and `plot_utils.py` modules from the original Dash app for:

- Statistical tests requiring scipy (not available in the browser)
- Publication-quality vector export at 4x resolution via Kaleido
- Batch processing across all reaction types and reactant categories
