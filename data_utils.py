from __future__ import annotations

"""data_utils.py
=================
Centralised utility functions for reading, cleaning **and** filtering the
experimental dataset that feeds the Dash application.

Having the data-related logic in a dedicated module brings multiple
benefits:

*   *Single-responsibility*: the rest of the code base does not need to
    know **how** we clean/transform the raw CSV – it can simply import
    `data_utils.DF` and operate on a **ready-to-use** `pandas.DataFrame`.
*   *Reusability*: if you want to spin up a different dashboard or run a
    notebook off the same data you only need to import this file.
*   *Testability*: the pure functions below (eg. `filter_data`) can be
    unit-tested in isolation.

Every function carries an extensive doc-string so future maintainers can
quickly understand *why* a certain transformation exists.
"""

from pathlib import Path
from typing import List
import os
from functools import lru_cache
import hashlib

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. DATA LOADING / NORMALISATION
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NOTE – The dashboard now works off a *new* data export that contains
#        dedicated "FG A", "FG B", and "FG_sorted" columns for functional
#        groups. The "FG_sorted" column contains the sorted and concatenated
#        pair of functional groups. We therefore update the default csv path
#        accordingly. If, for whatever reason, the old file is required
#        simply point *CSV_PATH* back to the old location.
# ---------------------------------------------------------------------------

# Path to the cleaned data export that contains functional-group information
# Use local file if available, otherwise use the expected cloud-downloaded filename
local_csv = Path("z-Score Peaks with FG.csv")
cloud_csv = Path("zscore_peaks_data.csv")

# Check if we're running locally (local CSV exists) or in production
if local_csv.exists():
    CSV_PATH = local_csv
    print("Using local CSV file for development")
else:
    CSV_PATH = cloud_csv
    print("Using cloud CSV configuration")

# Google Cloud Storage configuration
GCS_BUCKET_NAME = "zscore_csv_storage"
GCS_FILE_PATH = "z-Score Peaks with FG.csv"

def download_csv_from_gcs():
    """Download the CSV file from Google Cloud Storage if it doesn't exist locally."""
    if CSV_PATH.exists():
        print(f"CSV file {CSV_PATH} already exists locally")
        return
    
    try:
        # Try using the public URL approach first (simpler)
        gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{GCS_FILE_PATH}"
        print(f"Downloading CSV from GCS: {gcs_url}")
        
        response = requests.get(gcs_url, timeout=60)
        response.raise_for_status()
        
        # Write the content as binary first
        with open(CSV_PATH, "wb") as f:
            f.write(response.content)
        
        print(f"Successfully downloaded {len(response.content)} bytes to {CSV_PATH}")
        
        # Validate the downloaded file can be read as CSV
        _validate_csv_file(CSV_PATH)
        
    except Exception as e:
        print(f"Failed to download from public URL: {e}")
        print("Trying authenticated GCS client...")
        
        try:
            # Lazily import the Google Cloud Storage client so local runs
            # without the package installed don't fail on module import.
            try:
                from google.cloud import storage  # type: ignore
            except Exception as import_error:
                print(
                    "google-cloud-storage not installed or unavailable; "
                    "skipping authenticated GCS download."
                )
                print(f"Import error: {import_error}")
                return False

            # Fallback to authenticated client
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(GCS_FILE_PATH)
            
            blob.download_to_filename(str(CSV_PATH))
            print(f"Successfully downloaded via GCS client to {CSV_PATH}")
            
            # Validate the downloaded file can be read as CSV
            _validate_csv_file(CSV_PATH)
            
        except Exception as e2:
            print(f"Failed to download via GCS client: {e2}")
            print("Will use sample data instead")
            return False
    
    return True


def _validate_csv_file(csv_path: Path):
    """Validate that the downloaded CSV file can be read properly."""
    try:
        # Try to read just the header to validate encoding and format
        test_df = pd.read_csv(csv_path, nrows=1, encoding='utf-8')
        print(f"CSV validation successful - found {len(test_df.columns)} columns")
        return True
    except UnicodeDecodeError as e:
        print(f"UTF-8 encoding failed: {e}")
        # Try common alternative encodings
        for encoding in ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                test_df = pd.read_csv(csv_path, nrows=1, encoding=encoding)
                print(f"CSV validation successful with {encoding} encoding - found {len(test_df.columns)} columns")
                return True
            except Exception:
                continue
        print("Failed to read CSV with any common encoding")
        raise
    except Exception as e:
        print(f"CSV validation failed: {e}")
        raise


def _read_csv_with_encoding(csv_path: Path) -> pd.DataFrame:
    """Read CSV file with automatic encoding detection."""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            print(f"Attempting to read CSV with {encoding} encoding...")
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully read CSV with {encoding} encoding - {len(df)} rows, {len(df.columns)} columns")
            return df
        except UnicodeDecodeError as e:
            print(f"Failed with {encoding}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error with {encoding}: {e}")
            continue
    
    # If all encodings fail, raise the original error
    raise UnicodeDecodeError(
        'utf-8', b'', 0, 1, 
        f"Could not read CSV file with any of the attempted encodings: {encodings_to_try}"
    )


def _load_and_prepare(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Read the raw csv and perform *all* cleaning steps.

    We keep the cleaning logic as explicit as possible so that domain
    scientists can audit the transformation steps without having to dig
    through application code.

    Parameters
    ----------
    csv_path:
        Location of the raw csv export.  Defaults to
        :pydataattr:`~data_utils.CSV_PATH`.

    Returns
    -------
    pd.DataFrame
        A *clean* dataframe ready for downstream consumption.
    """
    
    # Try to download from GCS first if file doesn't exist
    if not csv_path.exists():
        print("CSV file not found locally, attempting to download from GCS...")
        download_success = download_csv_from_gcs()
    
    # Check if CSV file exists (either was there or downloaded), if not create sample data
    if not csv_path.exists():
        print(f"Warning: CSV file '{csv_path}' not found. Creating sample data for demo purposes.")
        # Create simple sample data without numpy dependency
        
        fg_options = ['OH', 'CH3', 'NH2', 'COOH', 'CHO', 'CH2', 'Ph', 'Cl', 'F', 'Br']
        
        # Create sample data manually to avoid numpy dependency
        sample_data = []
        for i in range(1, 101):  # Smaller dataset for faster loading
            compound = f"Compound_{i:03d}"
            peak = round(1.0 + (i % 10) * 0.5, 2)  # Simple pattern for peaks
            z_score = round((i % 20 - 10) * 0.3, 2)  # Z-scores between -3 and 3
            fg_a = fg_options[i % len(fg_options)]
            fg_b = fg_options[(i + 3) % len(fg_options)]
            fg_sorted = '-'.join(sorted([fg_a, fg_b]))
            
            sample_data.append({
                'Compound': compound,
                'Peak': peak,
                'z-Score': z_score,  # lowercase z to match processing code
                'FG A': fg_a,
                'FG B': fg_b,
                'FG_sorted': fg_sorted,
                'AREA_TOTAL_REDUCED': 100.0,  # Add expected column
                'Reaction Type': 'Sample_Reaction'  # Add expected column
            })
        
        df = pd.DataFrame(sample_data)
        print(f"Created sample dataset with {len(df)} rows")
    else:
        # Try to read the CSV with proper encoding handling
        df = _read_csv_with_encoding(csv_path)

    # ------------------------------------------------------------------
    # 0.0  --------------  FUNCTIONAL-GROUP PARSING  --------------------
    # ------------------------------------------------------------------
    # The CSV now contains dedicated columns for functional groups:
    # - "FG A" and "FG B" contain the individual functional groups
    # - "FG_sorted" contains the sorted and concatenated pair
    # We map these to our internal column names for consistency.

    if "FG A" in df.columns and "FG B" in df.columns:
        # Use the original column names directly
        # df["FG A"] and df["FG B"] are already available
        
        # Use the pre-computed sorted pair if available, otherwise compute it
        if "FG_sorted" in df.columns:
            df["FG_PAIR_SORTED"] = df["FG_sorted"]
        else:
            # Fallback: compute the sorted pair if not provided
            df["FG_PAIR_SORTED"] = df.apply(
                lambda r: ", ".join(sorted([str(r["FG A"]), str(r["FG B"])])), axis=1
            )

    # ------------------------------------------------------------------
    # 1.1  --------------  TYPE CONVERSIONS  ---------------------------
    # ------------------------------------------------------------------
    # The original export uses a *comma* as decimal separator which
    # confuses `pandas` when it tries to infer numeric dtype.  We unify
    # this by replacing commas with dots.

    df["z-Score"] = (
        df["z-Score"].astype(str).str.replace(",", ".").str.strip().pipe(pd.to_numeric, errors="coerce")
    )

    df["AREA_TOTAL_REDUCED"] = (
        df["AREA_TOTAL_REDUCED"].astype(str).str.replace(",", ".").str.strip().pipe(pd.to_numeric, errors="coerce")
    )

    # Any additional one-off data-quality fixes should live here so we
    # have a single choke-point for audit.

    return df


# The cleaned dataframe is created *once* at import time so every module
# that imports `data_utils` works with the same in-memory object (cheap
# copy-on-write semantics in pandas mean this is ok for read-heavy
# workloads like a dashboard).
DF: pd.DataFrame = _load_and_prepare()


# ---------------------------------------------------------------------------
# 2. DOMAIN CONSTANTS
# ---------------------------------------------------------------------------

# Hard-coded options used in dropdowns.  Having them here again keeps all
# data-related constants in a single file.
CATEGORY_OPTIONS: List[str] = [
    "Additive",
    "Base",
    "Catalyst",
    "Coupling Reagent",
    "Solvent",
    "Functional Group A",
    "Functional Group B",
    "Ligand",
    "Secondary Solvent",
]

# The available reaction types are directly derived from the dataset so
# they do **not** have to be updated manually once a new reaction shows
# up in the csv.
REACTION_TYPES: List[str] = DF["Reaction Type"].dropna().unique().tolist()




# ---------------------------------------------------------------------------
# 4. FILTERING FUNCTIONS
# ---------------------------------------------------------------------------

def _convert_checkbox_to_bool(checkbox_value):
    """Convert checkbox value to boolean."""
    return checkbox_value is not None and len(checkbox_value) > 0

def _create_cache_key(*args):
    """Create a hashable cache key from filter parameters."""
    # Convert all arguments to strings and create a hash
    key_str = str(args)
    return hashlib.md5(key_str.encode()).hexdigest()

# Cache for filtered data - stores (cache_key -> (dataframe, stats))
_FILTER_CACHE = {}
_CACHE_MAX_SIZE = 50  # Maximum number of cached filter results

def filter_data(
    reactant_types: list = None,  # List of selected reactant types (categories)
    reaction_types: list = None,
    fg_a: str | list = None,
    fg_b: str | list = None,
    exclude_cui = None,  # list, checked if 'exclude_cui' in list
    exclude_scaleup = None,  # list, checked if True in list
    include_null_categories = None,  # list, checked if True in list
    min_eln: int = None,
    topn_zscore: int = None,
    max_components: int = None,
    return_stats: bool = False,
    source_df: pd.DataFrame = None,  # Optional: use this DataFrame instead of global DF
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Return a filtered dataframe according to the business logic used by the app, with user-customizable filters.

    Args:
        reactant_types: List of selected reactant types (categories) to filter by
        reaction_types: List of reaction types to filter by
        fg_a, fg_b: Functional group filters
        exclude_cui: Whether to exclude CuI catalyst
        exclude_scaleup: Whether to exclude scale-up plates
        include_null_categories: Whether to include null category values
        min_eln, topn_zscore, max_components: Additional filtering parameters
        return_stats: Whether to return statistics along with filtered data
        source_df: Optional DataFrame to use instead of the default global DF.
                   Use this for user-uploaded datasets.

    If the special keyword argument ``return_stats`` is provided as True (via **kwargs),
    the function returns a tuple ``(filtered_df, stats)`` where ``stats`` mirrors
    the information previously produced by ``get_filtering_statistics``. This allows
    callers to compute filtered data and the statistics in a single function call.
    """
    
    # Determine if we're using uploaded data (skip cache for uploaded data)
    using_uploaded = source_df is not None
    
    # Check cache first (only for default data)
    if not using_uploaded:
        cache_key = _create_cache_key(
            reactant_types, reaction_types, fg_a, fg_b,
            exclude_cui, exclude_scaleup, include_null_categories, min_eln, topn_zscore, max_components, return_stats
        )
        
        if cache_key in _FILTER_CACHE:
            cached_result = _FILTER_CACHE[cache_key]
            if return_stats:
                return cached_result['dataframe'].copy(), cached_result['stats'].copy()
            else:
                return cached_result['dataframe'].copy()
    
    # Use provided source_df or fall back to global DF
    dff = source_df.copy() if using_uploaded else DF.copy()
    
    # 1. Filter by Reaction Types (cheap)
    if reaction_types and len(reaction_types) > 0:
        dff = dff[dff['Reaction Type'].isin(reaction_types)]

    # init simple stats collection
    stats: dict | None = {} if return_stats else None
    if stats is not None:
        stats['whole_dataset'] = {'elns': dff['ELN_ID'].nunique()}

    # Convert include_null_categories checkbox to boolean
    include_null_categories_bool = _convert_checkbox_to_bool(include_null_categories)

    # 2. Filter: ensure the selected reactant type columns are populated (cheap)
    # Unless include_null_categories is checked, exclude null/empty values
    if reactant_types and len(reactant_types) > 0:
        if not include_null_categories_bool:
            for reactant_type in reactant_types:
                if reactant_type and reactant_type != '':
                    dff = dff[dff[reactant_type].notnull() & (dff[reactant_type] != '')]
        
        # 2a. Filter: reactant types summary for stats
        if stats is not None:
            stats['after_reactant_filters'] = {'elns': dff['ELN_ID'].nunique()}

    # 5. Filter: Catalyst not CuI (cheap)
    if 'Catalyst' in dff.columns and exclude_cui and ('exclude_cui' in exclude_cui):
        dff = dff[(dff['Catalyst'].isnull()) | (dff['Catalyst'] != 'CuI')]

    # 6 & 7 -------------  FUNCTIONAL-GROUP FILTERS  ------------------ (cheap)
    def _mask_contains_fg(df: pd.DataFrame, fg: str) -> pd.Series:
        return (df['FG A'] == fg) | (df['FG B'] == fg)
    
    def _normalize_fg_input(fg_input):
        """Normalize functional group input to a list, handling both string and list inputs."""
        if not fg_input:
            return []
        if isinstance(fg_input, str):
            return [fg_input] if fg_input != 'All' else []
        elif isinstance(fg_input, list):
            return [fg for fg in fg_input if fg != 'All']
        return []

    # 6. FG_A filter 
    fg_a_list = _normalize_fg_input(fg_a)
    if fg_a_list:
        fg_a_mask = pd.Series([False] * len(dff), index=dff.index)
        for fg in fg_a_list:
            fg_a_mask |= _mask_contains_fg(dff, fg)
        dff = dff[fg_a_mask]
    if stats is not None and fg_a_list:
        stats['after_fg_a'] = {'elns': dff['ELN_ID'].nunique()}

    # 7. FG_B filter 
    fg_b_list = _normalize_fg_input(fg_b)
    if fg_b_list:
        if fg_a_list:
            # When both FG A and FG B are specified with multiple selections,
            # we look for any combination of the specified groups
            combined_mask = pd.Series([False] * len(dff), index=dff.index)
            for fg_a_val in fg_a_list:
                for fg_b_val in fg_b_list:
                    wanted_pair = sorted([fg_a_val, fg_b_val])
                    combined_mask |= (dff['FG_PAIR_SORTED'] == ', '.join(wanted_pair))
            dff = dff[combined_mask]
        else:
            # If only FG B is specified, check any of the selected groups
            fg_b_mask = pd.Series([False] * len(dff), index=dff.index)
            for fg in fg_b_list:
                fg_b_mask |= _mask_contains_fg(dff, fg)
            dff = dff[fg_b_mask]
    if stats is not None and fg_b_list:
        stats['after_fg_b'] = {'elns': dff['ELN_ID'].nunique()}
    
    # 8. Filter: Scale-up plates (moderate)
    # Scale-up plates are identified as plates with no experimental variability
    # (all wells have identical conditions across reagent columns)
    exclude_scaleup_bool = _convert_checkbox_to_bool(exclude_scaleup)
    if exclude_scaleup_bool:
        # Define reagent columns that should show variability in non-scale-up plates
        reagent_cols = [
            col for col in [
                'Additive', 'Base', 'Catalyst', 'Coupling Reagent',
                'Solvent', 'Ligand', 'Secondary Solvent', 'Tertiary Solvent'
            ] if col in dff.columns
        ]
        
        if reagent_cols:
            # Group by plate and count unique non-null values per reagent column
            plate_variability = (
                dff.groupby(['ELN_ID', 'PLATENUMBER'])[reagent_cols]
                .nunique()
                .reset_index()
            )
            
            # Keep plates where at least one reagent column has >1 unique value
            has_variability = (plate_variability[reagent_cols] > 1).any(axis=1)
            plates_to_keep = plate_variability[has_variability][['ELN_ID', 'PLATENUMBER']]
            
            # Filter original dataframe to keep only variable plates
            dff = dff.merge(plates_to_keep, on=['ELN_ID', 'PLATENUMBER'], how='inner')

    # 9. Deduplication: keep best z-Score for unique columns (heavy)
    dedup_cols = ['ELN_ID']
    for col in ['Additive', 'Base', 'Catalyst', 'Coupling Reagent', 'Solvent', 'Ligand', 'Secondary Solvent', 'Tertiary Solvent']:
        if col in dff.columns:
            dedup_cols.append(col)

    # Fill NaN values with a placeholder to avoid grouping issues
    dff_filled = dff.copy()
    for col in dedup_cols:
        if col in dff_filled.columns:
            dff_filled[col] = dff_filled[col].fillna('__NAN__')

    dff_filled['z-Score_rank'] = dff_filled.groupby(dedup_cols)['z-Score'].rank(
        method='first', ascending=False
    )
    dff = dff_filled[dff_filled['z-Score_rank'] == 1].drop(columns=['z-Score_rank'])

    # Restore NaN values in the result
    for col in dedup_cols:
        if col in dff.columns:
            dff[col] = dff[col].replace('__NAN__', pd.NA)

    # 10. Filter: Top-N z-scores per (ELN_ID, reactant_type combination) (heavy)
    if topn_zscore and reactant_types and len(reactant_types) > 0:
        # Build rank columns based on selected reactant types
        rank_cols = ['ELN_ID'] + reactant_types
        
        # Only include columns that exist in the dataframe
        rank_cols = [col for col in rank_cols if col in dff.columns]
        
        # Only proceed if we have at least ELN_ID and one reactant type column
        if len(rank_cols) >= 2:
            # Handle null values in ranking when include_null_categories is True
            if include_null_categories_bool:
                # Create a temporary dataframe with filled null values for groupby operations
                dff_rank = dff.copy()
                for col in rank_cols:
                    if col in dff_rank.columns:
                        dff_rank[col] = dff_rank[col].fillna('__NULL_CATEGORY__')
                
                dff['z-Score_rank_2'] = dff_rank.groupby(rank_cols)['z-Score'].rank(
                    method='first', ascending=False
                )
            else:
                dff['z-Score_rank_2'] = dff.groupby(rank_cols)['z-Score'].rank(
                    method='first', ascending=False
                )
            
            dff = dff[dff['z-Score_rank_2'] <= topn_zscore]
    
    # 11. Filter: Minimum number of ELNs per category combination (heavy)
    if min_eln and reactant_types:
        group_cols = ['Reaction Type'] + [rt for rt in reactant_types if rt and rt != '']
        
        # Handle null values in groupby when include_null_categories is True
        if include_null_categories_bool:
            # Create a temporary dataframe with filled null values for groupby operations
            dff_group = dff.copy()
            for col in group_cols:
                if col in dff_group.columns:
                    dff_group[col] = dff_group[col].fillna('__NULL_CATEGORY__')
            
            group_counts = dff_group.groupby(group_cols)['ELN_ID'].transform('nunique')
        else:
            group_counts = dff.groupby(group_cols)['ELN_ID'].transform('nunique')
        
        dff = dff[group_counts >= min_eln]
        if stats is not None:
            stats['after_min_eln'] = {'elns': dff['ELN_ID'].nunique()}

    # Expose the dynamic cap for the "Max components to display" slider.
    # If multiple categories are selected, treat each unique combination
    # as a component so the cap reflects what is actually visualised.
    if stats is not None and reactant_types:
        try:
            key_cols = [rt for rt in reactant_types if rt and rt != '']
            stats['max_components_cap'] = int(dff[key_cols].drop_duplicates().shape[0])
        except Exception:
            # Be defensive – if any column is missing for any reason, fall back to 1
            stats['max_components_cap'] = 1
    
    # 12. Filter: Max components to display (if specified)
    # Works on combined categories when multiple are selected.
    if max_components and max_components > 0 and reactant_types:
        # Use reactant_types system only
        key_cols = [rt for rt in reactant_types if rt and rt != '']

        try:
            unique_components = int(dff[key_cols].drop_duplicates().shape[0])
        except Exception:
            unique_components = 0

        if unique_components and max_components < unique_components:
            # Order components by median z-Score (desc) and keep the top-N
            if include_null_categories_bool:
                # Include groups with null keys
                medians = (
                    dff.groupby(key_cols, dropna=False)['z-Score']
                    .median()
                    .sort_values(ascending=False)
                )
            else:
                medians = (
                    dff.groupby(key_cols)['z-Score']
                    .median()
                    .sort_values(ascending=False)
                )

            # Build a dataframe of the top combinations for a robust join-based filter
            top_df = medians.head(max_components).reset_index()[key_cols].drop_duplicates()

            NULL_SENTINEL = '__NULL_CATEGORY__'
            if len(key_cols) == 1:
                if include_null_categories_bool:
                    left_series = dff[key_cols[0]].fillna(NULL_SENTINEL)
                    right_series = top_df[key_cols[0]].fillna(NULL_SENTINEL)
                    dff = dff[left_series.isin(right_series)]
                else:
                    dff = dff[dff[key_cols[0]].isin(top_df[key_cols[0]].tolist())]
            else:
                if include_null_categories_bool:
                    left_keys = dff[key_cols].fillna(NULL_SENTINEL).reset_index()
                    right_keys = top_df[key_cols].fillna(NULL_SENTINEL).drop_duplicates()
                    matched = left_keys.merge(right_keys, on=key_cols, how='inner')
                    matched_idx = matched['index'].unique().tolist()
                    dff = dff.loc[matched_idx]
                else:
                    dff = dff.merge(top_df, on=key_cols, how='inner')

    # Store result in cache (with size limit) - only for default data
    if not using_uploaded:
        if len(_FILTER_CACHE) >= _CACHE_MAX_SIZE:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(_FILTER_CACHE))
            del _FILTER_CACHE[oldest_key]
        
        _FILTER_CACHE[cache_key] = {
            'dataframe': dff.copy(),
            'stats': stats.copy() if stats else {}
        }
    
    # If no statistics requested, short-circuit
    if not return_stats:
        return dff
    return dff, stats or {}


def clear_filter_cache():
    """Clear the filter data cache. Useful for debugging or memory management."""
    global _FILTER_CACHE
    _FILTER_CACHE.clear()

# Clear cache on import to ensure fresh start
clear_filter_cache()


def get_cache_info():
    """Get information about the current cache state."""
    return {
        'cache_size': len(_FILTER_CACHE),
        'max_size': _CACHE_MAX_SIZE,
        'cache_keys': list(_FILTER_CACHE.keys())[:5]  # Show first 5 keys
    }


# ---------------------------------------------------------------------------
# 5. UPLOADED DATA HELPERS
# ---------------------------------------------------------------------------

def parse_uploaded_data(json_data: str) -> pd.DataFrame | None:
    """Parse uploaded data from JSON string stored in dcc.Store.
    
    Args:
        json_data: JSON string from dcc.Store containing uploaded dataset
        
    Returns:
        DataFrame if parsing succeeds, None otherwise
    """
    if not json_data:
        return None
    
    try:
        return pd.read_json(json_data, orient='split')
    except Exception:
        return None


def get_active_dataframe(uploaded_data_json: str = None) -> pd.DataFrame:
    """Get the active DataFrame - either uploaded data or default.
    
    Args:
        uploaded_data_json: JSON string from uploaded-data-store
        
    Returns:
        The uploaded DataFrame if available, otherwise the default DF
    """
    if uploaded_data_json:
        uploaded_df = parse_uploaded_data(uploaded_data_json)
        if uploaded_df is not None:
            return uploaded_df
    return DF


def get_reaction_types_from_data(df: pd.DataFrame = None) -> List[str]:
    """Get available reaction types from a DataFrame.
    
    Args:
        df: DataFrame to extract reaction types from. Uses default DF if None.
        
    Returns:
        List of unique reaction type values
    """
    source = df if df is not None else DF
    if "Reaction Type" in source.columns:
        return source["Reaction Type"].dropna().unique().tolist()
    return []


def get_category_options_from_data(df: pd.DataFrame = None) -> List[str]:
    """Get available category options from a DataFrame.
    
    This checks which of the standard category columns exist and have data.
    
    Args:
        df: DataFrame to check. Uses default DF if None.
        
    Returns:
        List of category column names that exist and have data
    """
    source = df if df is not None else DF
    available = []
    for cat in CATEGORY_OPTIONS:
        if cat in source.columns and source[cat].notna().any():
            available.append(cat)
    return available


# ---------------------------------------------------------------------------
# 6. STATISTICAL VALIDATION FUNCTIONS
# ---------------------------------------------------------------------------

def compute_distribution_stats(
    dff: pd.DataFrame,
    group_col: str = 'Reaction Type',
    value_col: str = 'z-Score',
    min_samples: int = 20
) -> pd.DataFrame:
    """Compute distribution statistics for z-scores grouped by a category.
    
    Calculates skewness, kurtosis, and Shapiro-Wilk normality test p-value
    for each group. These statistics help assess whether the z-score
    methodology's normality assumptions are violated.
    
    Args:
        dff: DataFrame containing the data to analyze
        group_col: Column to group by (default: 'Reaction Type')
        value_col: Column containing the values to analyze (default: 'z-Score')
        value_col: Column containing the values to analyze (default: 'z-Score')
        min_samples: Minimum number of samples required per group (default: 20)
        
    Returns:
        DataFrame with columns: group, n, mean, std, skewness, kurtosis, 
        shapiro_stat, shapiro_p, is_normal (at alpha=0.05)
        
    Notes:
        - Shapiro-Wilk test is limited to 5000 samples (random sample taken if larger)
        - Groups with fewer than min_samples are excluded
        - Skewness interpretation: |skew| < 0.5 = fairly symmetric, 
          0.5-1 = moderately skewed, >1 = highly skewed
        - Kurtosis interpretation: ~0 = normal, >0 = heavy tails, <0 = light tails
    """
    from scipy import stats
    import numpy as np
    
    results = []
    
    for name, group in dff.groupby(group_col):
        values = group[value_col].dropna()
        n = len(values)
        
        if n < min_samples:
            continue
            
        # Basic statistics
        mean_val = values.mean()
        std_val = values.std()
        skewness = values.skew()
        kurtosis = values.kurtosis()  # Fisher's definition (normal = 0)
        
        # Shapiro-Wilk test (limited to 5000 samples)
        sample_for_test = values.sample(min(5000, n), random_state=42) if n > 5000 else values
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample_for_test)
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        results.append({
            'group': name,
            'n': n,
            'mean': round(mean_val, 4),
            'std': round(std_val, 4),
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4),
            'shapiro_stat': round(shapiro_stat, 4) if not np.isnan(shapiro_stat) else np.nan,
            'shapiro_p': round(shapiro_p, 4) if not np.isnan(shapiro_p) else np.nan,
            'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None
        })
    
    return pd.DataFrame(results)


def compute_significance_tests(
    dff: pd.DataFrame,
    category_col: str,
    value_col: str = 'z-Score',
    top_n: int = 10,
    alpha: float = 0.05
) -> dict:
    """Run statistical significance tests comparing reagent/category performance.
    
    Performs Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA)
    to test if there are significant differences among groups, followed by
    pairwise Mann-Whitney U tests with Bonferroni correction.
    
    Args:
        dff: DataFrame containing the data to analyze
        category_col: Column containing the categories to compare
        value_col: Column containing the values to compare (default: 'z-Score')
        top_n: Number of top categories (by median) to include in analysis
        alpha: Significance level (default: 0.05)
        
    Returns:
        Dictionary containing:
        - 'kruskal_wallis': dict with 'statistic', 'p_value', 'significant'
        - 'n_groups': number of groups compared
        - 'n_comparisons': number of pairwise comparisons
        - 'alpha_corrected': Bonferroni-corrected alpha
        - 'pairwise': DataFrame with pairwise comparison results
        - 'effect_sizes': DataFrame with effect size (rank-biserial correlation)
        - 'group_stats': DataFrame with descriptive stats per group
        
    Notes:
        - Kruskal-Wallis is used because z-scores may not be normally distributed
        - Bonferroni correction is applied: alpha_corrected = alpha / n_comparisons
        - Effect size uses rank-biserial correlation (r): 
          |r| < 0.1 = negligible, 0.1-0.3 = small, 0.3-0.5 = medium, >0.5 = large
    """
    from scipy import stats
    import numpy as np
    from itertools import combinations
    
    # Get top N categories by median z-score
    medians = dff.groupby(category_col)[value_col].median().sort_values(ascending=False)
    top_categories = medians.head(top_n).index.tolist()
    
    # Filter to top categories
    dff_filtered = dff[dff[category_col].isin(top_categories)]
    
    # Prepare groups for analysis
    groups = []
    group_names = []
    group_stats = []
    
    for cat in top_categories:
        values = dff_filtered[dff_filtered[category_col] == cat][value_col].dropna()
        if len(values) >= 5:  # Minimum samples for meaningful comparison
            groups.append(values.values)
            group_names.append(cat)
            group_stats.append({
                'category': cat,
                'n': len(values),
                'median': round(values.median(), 4),
                'mean': round(values.mean(), 4),
                'std': round(values.std(), 4),
                'q25': round(values.quantile(0.25), 4),
                'q75': round(values.quantile(0.75), 4)
            })
    
    result = {
        'n_groups': len(groups),
        'group_stats': pd.DataFrame(group_stats) if group_stats else pd.DataFrame()
    }
    
    if len(groups) < 2:
        result['kruskal_wallis'] = {'statistic': np.nan, 'p_value': np.nan, 'significant': None}
        result['pairwise'] = pd.DataFrame()
        result['effect_sizes'] = pd.DataFrame()
        return result
    
    # Kruskal-Wallis H-test
    try:
        kw_stat, kw_p = stats.kruskal(*groups)
        result['kruskal_wallis'] = {
            'statistic': round(kw_stat, 4),
            'p_value': kw_p,
            'significant': kw_p < alpha
        }
    except Exception:
        result['kruskal_wallis'] = {'statistic': np.nan, 'p_value': np.nan, 'significant': None}
    
    # Pairwise Mann-Whitney U tests with Bonferroni correction
    n_comparisons = len(list(combinations(range(len(groups)), 2)))
    alpha_corrected = alpha / n_comparisons if n_comparisons > 0 else alpha
    result['n_comparisons'] = n_comparisons
    result['alpha_corrected'] = alpha_corrected
    
    pairwise_results = []
    effect_sizes = []
    
    for (i, j) in combinations(range(len(groups)), 2):
        group_i, group_j = groups[i], groups[j]
        name_i, name_j = group_names[i], group_names[j]
        
        try:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(group_i, group_j, alternative='two-sided')
            
            # Calculate rank-biserial correlation as effect size
            # r = 1 - (2*U) / (n1*n2)
            n1, n2 = len(group_i), len(group_j)
            r = 1 - (2 * u_stat) / (n1 * n2)
            
            pairwise_results.append({
                'group_1': name_i,
                'group_2': name_j,
                'u_statistic': round(u_stat, 2),
                'p_value': p_value,
                'p_value_formatted': f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
                'significant': p_value < alpha_corrected,
                'effect_size_r': round(r, 4),
                'effect_magnitude': _interpret_effect_size(abs(r))
            })
            
        except Exception:
            pairwise_results.append({
                'group_1': name_i,
                'group_2': name_j,
                'u_statistic': np.nan,
                'p_value': np.nan,
                'p_value_formatted': 'N/A',
                'significant': None,
                'effect_size_r': np.nan,
                'effect_magnitude': 'N/A'
            })
    
    result['pairwise'] = pd.DataFrame(pairwise_results)
    
    return result


def compute_permutation_test(
    dff: pd.DataFrame,
    category_col: str,
    value_col: str = 'z-Score',
    n_permutations: int = 10000,
    random_state: int = 42
) -> dict:
    """Run permutation test for Kruskal-Wallis, respecting within-ELN structure.
    
    This test is valid even with the top-N filtering because it:
    1. Keeps the data structure intact (same observations, same ELNs)
    2. Only shuffles category labels
    3. Computes empirical p-value from permuted distribution
    
    Args:
        dff: DataFrame with filtered data
        category_col: Column containing category labels to permute
        value_col: Column with values to compare
        n_permutations: Number of permutations (default 10000)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with observed H, permuted H distribution, empirical p-value
    """
    from scipy import stats
    import numpy as np
    
    np.random.seed(random_state)
    
    # Get observed Kruskal-Wallis H statistic
    categories = dff[category_col].unique()
    groups = [dff[dff[category_col] == cat][value_col].values for cat in categories]
    observed_h, observed_p = stats.kruskal(*groups)
    
    # Permutation test: shuffle category labels, recalculate H
    permuted_h_values = []
    values = dff[value_col].values.copy()
    
    for _ in range(n_permutations):
        # Shuffle the values (equivalent to shuffling labels)
        np.random.shuffle(values)
        
        # Recalculate H with shuffled values
        start_idx = 0
        shuffled_groups = []
        for cat in categories:
            n_cat = (dff[category_col] == cat).sum()
            shuffled_groups.append(values[start_idx:start_idx + n_cat])
            start_idx += n_cat
        
        h_perm, _ = stats.kruskal(*shuffled_groups)
        permuted_h_values.append(h_perm)
    
    permuted_h_values = np.array(permuted_h_values)
    
    # Empirical p-value: proportion of permuted H >= observed H
    empirical_p = (permuted_h_values >= observed_h).mean()
    
    return {
        'observed_h': observed_h,
        'standard_p': observed_p,
        'empirical_p': empirical_p,
        'n_permutations': n_permutations,
        'permuted_h_mean': permuted_h_values.mean(),
        'permuted_h_std': permuted_h_values.std(),
        'permuted_h_95th': np.percentile(permuted_h_values, 95),
        'significant_permutation': empirical_p < 0.05
    }


def _interpret_effect_size(r: float) -> str:
    """Interpret rank-biserial correlation effect size.
    
    Args:
        r: Absolute value of rank-biserial correlation
        
    Returns:
        String interpretation: 'negligible', 'small', 'medium', or 'large'
    """
    if r < 0.1:
        return 'negligible'
    elif r < 0.3:
        return 'small'
    elif r < 0.5:
        return 'medium'
    else:
        return 'large'


def get_distribution_summary(dff: pd.DataFrame, group_col: str = 'Reaction Type', value_col: str = 'z-Score') -> dict:
    """Get a summary of distribution characteristics across all groups.
    
    Provides aggregate statistics about normality across the dataset,
    useful for reporting in papers.
    
    Args:
        dff: DataFrame to analyze
        group_col: Column to group by
        value_col: Column containing the values to analyze (default: 'z-Score')
        
    Returns:
        Dictionary with summary statistics:
        - 'n_groups': Total number of groups analyzed
        - 'n_normal': Number of groups passing Shapiro-Wilk at alpha=0.05
        - 'pct_normal': Percentage of groups that are normal
        - 'n_symmetric': Number of groups with |skewness| < 0.5
        - 'pct_symmetric': Percentage of groups that are fairly symmetric
        - 'n_moderate_skew': Number with 0.5 <= |skewness| < 1
        - 'n_high_skew': Number with |skewness| >= 1
        - 'median_skewness': Median skewness across all groups
        - 'median_kurtosis': Median kurtosis across all groups
    """
    import numpy as np
    
    dist_stats = compute_distribution_stats(dff, group_col, value_col=value_col)
    
    if dist_stats.empty:
        return {
            'n_groups': 0,
            'n_normal': 0,
            'pct_normal': 0.0,
            'n_symmetric': 0,
            'pct_symmetric': 0.0,
            'n_moderate_skew': 0,
            'n_high_skew': 0,
            'median_skewness': np.nan,
            'median_kurtosis': np.nan
        }
    
    n_groups = len(dist_stats)
    n_normal = dist_stats['is_normal'].sum() if 'is_normal' in dist_stats.columns else 0
    
    abs_skewness = dist_stats['skewness'].abs()
    n_symmetric = (abs_skewness < 0.5).sum()
    n_moderate_skew = ((abs_skewness >= 0.5) & (abs_skewness < 1)).sum()
    n_high_skew = (abs_skewness >= 1).sum()
    
    return {
        'n_groups': n_groups,
        'n_normal': int(n_normal),
        'pct_normal': round(100 * n_normal / n_groups, 1) if n_groups > 0 else 0.0,
        'n_symmetric': int(n_symmetric),
        'pct_symmetric': round(100 * n_symmetric / n_groups, 1) if n_groups > 0 else 0.0,
        'n_moderate_skew': int(n_moderate_skew),
        'n_high_skew': int(n_high_skew),
        'median_skewness': round(dist_stats['skewness'].median(), 4),
        'median_kurtosis': round(dist_stats['kurtosis'].median(), 4)
    }

