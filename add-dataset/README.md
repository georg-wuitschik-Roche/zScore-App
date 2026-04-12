# Adding a New Dataset Version

Drop a CSV file into this folder and commit it. A pre-commit hook will automatically:

1. Assign the next version number (v1, v2, v3, ...)
2. Convert the CSV to a compressed Parquet file in `frontend/public/data/`
3. Generate the dropdown index for instant UI interactivity
4. Update `frontend/public/data/versions.json` with the new entry and today's date
5. Remove the source CSV from this folder (it's now stored as Parquet)
6. Stage all generated files for the commit

## Versioning convention

- **New integer** (v3, v4, ...) — substantially more data (e.g. 70k → 90k rows, new reaction types). Use the standard flow below.
- **Dotted minor** (v2.1, v2.2, ...) — data quality improvements (filtering, cleaning, corrections) without substantially increasing row count. Replaces its parent. Use the manual flow.

## Standard flow (new major version)

```bash
# 1. Copy your CSV into this folder
cp /path/to/my-new-data.csv add-dataset/

# 2. Stage and commit — the hook does the rest
git add add-dataset/my-new-data.csv
git commit -m "Add new dataset version"
```

## Manual flow (minor version / replace parent)

```bash
# Generate v2.1 from a CSV, removing v2 from the manifest
python3 scripts/version_dataset.py --version v2.1 --replace v2 path/to/data.csv
```

## Requirements

The CSV must contain these columns:

| Column | Description |
|--------|-------------|
| `ELN_ID` | Electronic lab notebook identifier |
| `PLATENUMBER` | Plate number |
| `Coordinate` | Well coordinate |
| `AREA_TOTAL_REDUCED` | Total reduced area (numeric) |
| `Base` | Base reagent |
| `Catalyst` | Catalyst |
| `Solvent` | Solvent |
| `Ligand` | Ligand |
| `Reaction Type` | Type of reaction |
| `FG A` | Functional group A |
| `FG B` | Functional group B |
| `FG_sorted` | Sorted functional group pair |
| `z-Score` | z-Score value (numeric) |

Optional columns: `Additive`, `Coupling Reagent`, `Secondary Solvent`, `Tertiary Solvent`, `output_column`.

## What happens in the app

- The newest version is loaded by default
- Users can switch between versions via the settings menu (gear icon)
- Switching versions preserves the current filter selections
- The active version is stored in the URL (`&ver=v2`) for shareable links

## Manual run

You can also run the versioning script manually without committing:

```bash
python3 scripts/version_dataset.py                                     # scan add-dataset/
python3 scripts/version_dataset.py path/to/data.csv                    # specific file
python3 scripts/version_dataset.py --version v3.1 --replace v3 data.csv  # minor version
```
