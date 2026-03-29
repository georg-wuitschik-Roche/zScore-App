# Adding a New Dataset Version

Drop a CSV file into this folder and commit it. A pre-commit hook will automatically:

1. Assign the next version number (v1, v2, v3, ...)
2. Convert the CSV to a compressed Parquet file in `frontend/public/data/`
3. Generate the dropdown index for instant UI interactivity
4. Update `frontend/public/data/versions.json` with the new entry and today's date
5. Remove the source CSV from this folder (it's now stored as Parquet)
6. Stage all generated files for the commit

## Steps

```bash
# 1. Copy your CSV into this folder
cp /path/to/my-new-data.csv add-dataset/

# 2. Stage and commit — the hook does the rest
git add add-dataset/my-new-data.csv
git commit -m "Add new dataset version"
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
python3 scripts/version_dataset.py
```
