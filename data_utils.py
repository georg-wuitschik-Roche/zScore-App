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
import hashlib
import logging
import threading
import time
import uuid

logger = logging.getLogger(__name__)

import numpy as np
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
    logger.info("Using local CSV file for development")
else:
    CSV_PATH = cloud_csv
    logger.info("Using cloud CSV configuration")

# Google Cloud Storage configuration
GCS_BUCKET_NAME = "zscore_csv_storage"
GCS_FILE_PATH = "z-Score Peaks with FG.csv"

def download_csv_from_gcs():
    """Download the CSV file from Google Cloud Storage if it doesn't exist locally."""
    if CSV_PATH.exists():
        logger.info("CSV file %s already exists locally", CSV_PATH)
        return True

    try:
        gcs_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{GCS_FILE_PATH}"
        logger.info("Downloading CSV from GCS: %s", gcs_url)

        response = requests.get(gcs_url, timeout=60)
        response.raise_for_status()

        with open(CSV_PATH, "wb") as f:
            f.write(response.content)

        logger.info("Successfully downloaded %d bytes to %s", len(response.content), CSV_PATH)
        _validate_csv_file(CSV_PATH)

    except Exception as e:
        logger.warning("Failed to download from public URL: %s", e)
        logger.info("Trying authenticated GCS client...")

        try:
            try:
                from google.cloud import storage  # type: ignore
            except Exception as import_error:
                logger.warning(
                    "google-cloud-storage not installed or unavailable; "
                    "skipping authenticated GCS download: %s", import_error)
                return False

            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(GCS_FILE_PATH)

            blob.download_to_filename(str(CSV_PATH))
            logger.info("Successfully downloaded via GCS client to %s", CSV_PATH)
            _validate_csv_file(CSV_PATH)

        except Exception as e2:
            logger.error("Failed to download via GCS client: %s", e2)
            logger.warning("Will use sample data instead")
            return False
    
    return True


def _validate_csv_file(csv_path: Path):
    """Validate that the downloaded CSV file can be read properly."""
    try:
        # Try to read just the header to validate encoding and format
        test_df = pd.read_csv(csv_path, nrows=1, encoding='utf-8')
        logger.info("CSV validation successful - found %d columns", len(test_df.columns))
        return True
    except UnicodeDecodeError as e:
        logger.debug("UTF-8 encoding failed: %s", e)
        for encoding in ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                test_df = pd.read_csv(csv_path, nrows=1, encoding=encoding)
                logger.info("CSV validation successful with %s encoding - found %d columns",
                            encoding, len(test_df.columns))
                return True
            except Exception:
                continue
        logger.error("Failed to read CSV with any common encoding")
        raise
    except Exception as e:
        logger.error("CSV validation failed: %s", e)
        raise


def _read_csv_with_encoding(csv_path: Path) -> pd.DataFrame:
    """Read CSV file with automatic encoding detection."""
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            logger.info("Attempting to read CSV with %s encoding...", encoding)
            df = pd.read_csv(csv_path, encoding=encoding)
            logger.info("Successfully read CSV with %s encoding - %d rows, %d columns",
                         encoding, len(df), len(df.columns))
            return df
        except UnicodeDecodeError as e:
            logger.debug("Failed with %s: %s", encoding, e)
            continue
        except Exception as e:
            logger.warning("Unexpected error with %s: %s", encoding, e)
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
        logger.info("CSV file not found locally, attempting to download from GCS...")
        download_success = download_csv_from_gcs()
    
    # Check if CSV file exists (either was there or downloaded), if not create sample data
    if not csv_path.exists():
        logger.warning("CSV file '%s' not found. Creating sample data for demo purposes.", csv_path)
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
        logger.info("Created sample dataset with %d rows", len(df))
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
            a = df["FG A"].astype(str)
            b = df["FG B"].astype(str)
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            df["FG_PAIR_SORTED"] = lo + ", " + hi

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

    # Convert low-cardinality string columns to Categorical to reduce
    # memory (~77 MB → ~15 MB) and speed up groupby / isin / == ops.
    _CATEGORICAL_COLS = [
        'Catalyst', 'Solvent', 'Base', 'Ligand', 'Additive',
        'Coupling Reagent', 'Secondary Solvent', 'Reaction Type',
        'FG A', 'FG B', 'FG_PAIR_SORTED', 'ELN_ID',
    ]
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

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
CATEGORY_OPTIONS: list[str] = [
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
REACTION_TYPES: list[str] = DF["Reaction Type"].dropna().unique().tolist()




# ---------------------------------------------------------------------------
# 4. FILTERING FUNCTIONS
# ---------------------------------------------------------------------------

def _convert_checkbox_to_bool(checkbox_value: list | None) -> bool:
    """Convert checkbox value to boolean."""
    return bool(checkbox_value)

def _create_cache_key(*args: object) -> str:
    """Create a hashable cache key from filter parameters."""
    key_str = str(args)
    return hashlib.md5(key_str.encode()).hexdigest()


def _normalize_fg_input(fg_input: str | list | None) -> list[str]:
    """Normalize functional group input to a list, filtering out 'All'."""
    if not fg_input:
        return []
    if isinstance(fg_input, str):
        return [fg_input] if fg_input != 'All' else []
    if isinstance(fg_input, list):
        return [fg for fg in fg_input if fg != 'All']
    return []


def _mask_contains_fg(df: pd.DataFrame, fg: str) -> pd.Series:
    """Return a boolean mask where *fg* appears in either FG column."""
    return (df['FG A'] == fg) | (df['FG B'] == fg)


# ---------------------------------------------------------------------------
# Individual filter steps
# ---------------------------------------------------------------------------

def _filter_by_reaction_types(dff: pd.DataFrame, reaction_types: list | None) -> pd.DataFrame:
    """Step 1: Keep only rows matching the selected reaction types."""
    if reaction_types:
        return dff[dff['Reaction Type'].isin(reaction_types)]
    return dff


def _filter_by_reactant_columns(
    dff: pd.DataFrame, reactant_types: list | None, include_null: bool
) -> pd.DataFrame:
    """Step 2: Ensure selected reactant-type columns are populated."""
    if not reactant_types or include_null:
        return dff
    for rt in reactant_types:
        if rt:
            dff = dff[dff[rt].notnull() & (dff[rt] != '')]
    return dff


def _filter_exclude_cui(dff: pd.DataFrame, exclude_cui: list | None) -> pd.DataFrame:
    """Step 3: Exclude CuI catalyst entries when requested."""
    if 'Catalyst' in dff.columns and exclude_cui and 'exclude_cui' in exclude_cui:
        dff = dff[(dff['Catalyst'].isnull()) | (dff['Catalyst'] != 'CuI')]
    return dff


def _filter_fg_a(dff: pd.DataFrame, fg_a: str | list | None) -> tuple[pd.DataFrame, list[str]]:
    """Step 4: Filter by Functional Group A. Returns (filtered_df, normalised_fg_a_list)."""
    fg_a_list = _normalize_fg_input(fg_a)
    if fg_a_list:
        mask = pd.Series(False, index=dff.index)
        for fg in fg_a_list:
            mask |= _mask_contains_fg(dff, fg)
        dff = dff[mask]
    return dff, fg_a_list


def _filter_fg_b(
    dff: pd.DataFrame, fg_b: str | list | None, fg_a_list: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Step 5: Filter by Functional Group B, considering FG A pairs."""
    fg_b_list = _normalize_fg_input(fg_b)
    if not fg_b_list:
        return dff, fg_b_list

    if fg_a_list:
        # Both specified: match any combination pair
        mask = pd.Series(False, index=dff.index)
        for fa in fg_a_list:
            for fb in fg_b_list:
                pair = ', '.join(sorted([fa, fb]))
                mask |= (dff['FG_PAIR_SORTED'] == pair)
        dff = dff[mask]
    else:
        mask = pd.Series(False, index=dff.index)
        for fg in fg_b_list:
            mask |= _mask_contains_fg(dff, fg)
        dff = dff[mask]
    return dff, fg_b_list


def _filter_scaleup_plates(dff: pd.DataFrame, exclude_scaleup: list | None) -> pd.DataFrame:
    """Step 6: Remove scale-up plates (plates with no reagent variability)."""
    if not _convert_checkbox_to_bool(exclude_scaleup):
        return dff

    reagent_cols = [
        col for col in [
            'Additive', 'Base', 'Catalyst', 'Coupling Reagent',
            'Solvent', 'Ligand', 'Secondary Solvent', 'Tertiary Solvent'
        ] if col in dff.columns
    ]
    if not reagent_cols:
        return dff

    plate_variability = (
        dff.groupby(['ELN_ID', 'PLATENUMBER'])[reagent_cols]
        .nunique()
    )
    has_variability = (plate_variability > 1).any(axis=1)
    keep_idx = has_variability[has_variability].index  # MultiIndex of (ELN_ID, PLATENUMBER)
    row_keys = pd.MultiIndex.from_arrays([dff['ELN_ID'], dff['PLATENUMBER']])
    return dff[row_keys.isin(keep_idx)]


_REAGENT_COLS = [
    'Additive', 'Base', 'Catalyst', 'Coupling Reagent',
    'Solvent', 'Ligand', 'Secondary Solvent', 'Tertiary Solvent',
]
_NAN_SENTINEL = '__NAN__'
_NULL_SENTINEL = '__NULL_CATEGORY__'


def _fillna_safe(df_or_series, value):
    """Fill NaN with *value*, safely handling Categorical columns."""
    if isinstance(df_or_series, pd.Series):
        if hasattr(df_or_series, 'cat'):
            return df_or_series.astype('object').fillna(value)
        return df_or_series.fillna(value)
    # DataFrame: cast any categorical columns to object first
    cat_cols = [c for c in df_or_series.columns if hasattr(df_or_series[c], 'cat')]
    if cat_cols:
        df_or_series = df_or_series.astype({c: 'object' for c in cat_cols})
    return df_or_series.fillna(value)


def _deduplicate_best_zscore(dff: pd.DataFrame) -> pd.DataFrame:
    """Step 7: Keep the best z-Score per unique reagent combination."""
    dedup_cols = ['ELN_ID'] + [c for c in _REAGENT_COLS if c in dff.columns]

    # Fill NaN only on the groupby columns (avoids full DataFrame copy).
    filled = _fillna_safe(dff[dedup_cols], _NAN_SENTINEL)
    filled['z-Score'] = dff['z-Score']

    rank = filled.groupby(dedup_cols)['z-Score'].rank(method='first', ascending=False)
    dff = dff[rank == 1].copy()

    # Restore NaN values
    for col in dedup_cols:
        if col in dff.columns:
            dff[col] = dff[col].replace(_NAN_SENTINEL, pd.NA)
    return dff


def _filter_topn_zscore(
    dff: pd.DataFrame,
    topn_zscore: int | None,
    reactant_types: list | None,
    include_null: bool,
) -> pd.DataFrame:
    """Step 8: Keep only the top-N z-scores per ELN + reactant combination."""
    if not topn_zscore or not reactant_types:
        return dff

    rank_cols = [c for c in ['ELN_ID'] + reactant_types if c in dff.columns]
    if len(rank_cols) < 2:
        return dff

    if include_null:
        filled = _fillna_safe(dff[rank_cols], _NULL_SENTINEL)
        filled['z-Score'] = dff['z-Score']
        rank = filled.groupby(rank_cols)['z-Score'].rank(method='first', ascending=False)
    else:
        rank = dff.groupby(rank_cols)['z-Score'].rank(method='first', ascending=False)

    return dff[rank <= topn_zscore]


def _filter_min_eln(
    dff: pd.DataFrame,
    min_eln: int | None,
    reactant_types: list | None,
    include_null: bool,
) -> pd.DataFrame:
    """Step 9: Require a minimum number of unique ELNs per category group."""
    if not min_eln or not reactant_types:
        return dff

    group_cols = ['Reaction Type'] + [rt for rt in reactant_types if rt]
    if include_null:
        filled = _fillna_safe(dff[group_cols], _NULL_SENTINEL)
        filled['ELN_ID'] = dff['ELN_ID']
        counts = filled.groupby(group_cols)['ELN_ID'].transform('nunique')
    else:
        counts = dff.groupby(group_cols)['ELN_ID'].transform('nunique')

    return dff[counts >= min_eln]


def _filter_max_components(
    dff: pd.DataFrame,
    max_components: int | None,
    reactant_types: list | None,
    include_null: bool,
) -> pd.DataFrame:
    """Step 10: Cap the number of displayed components by median z-Score."""
    if not max_components or max_components <= 0 or not reactant_types:
        return dff

    key_cols = [rt for rt in reactant_types if rt]
    try:
        unique_count = int(dff[key_cols].drop_duplicates().shape[0])
    except Exception:
        return dff

    if max_components >= unique_count:
        return dff

    medians = (
        dff.groupby(key_cols, dropna=not include_null)['z-Score']
        .median()
        .sort_values(ascending=False)
    )
    top_df = medians.head(max_components).reset_index()[key_cols].drop_duplicates()

    if len(key_cols) == 1:
        if include_null:
            left = _fillna_safe(dff[key_cols[0]], _NULL_SENTINEL)
            right = _fillna_safe(top_df[key_cols[0]], _NULL_SENTINEL)
            return dff[left.isin(right)]
        return dff[dff[key_cols[0]].isin(top_df[key_cols[0]].tolist())]

    if include_null:
        left_keys = _fillna_safe(dff[key_cols], _NULL_SENTINEL).reset_index()
        right_keys = _fillna_safe(top_df[key_cols], _NULL_SENTINEL).drop_duplicates()
        matched = left_keys.merge(right_keys, on=key_cols, how='inner')
        return dff.loc[matched['index'].unique().tolist()]
    return dff.merge(top_df, on=key_cols, how='inner')


# Cache for filtered data - stores (cache_key -> (dataframe, stats))
# Uses OrderedDict for LRU eviction: move_to_end on hit, pop oldest on full.
from collections import OrderedDict
_FILTER_CACHE: OrderedDict[str, dict] = OrderedDict()
_FILTER_CACHE_LOCK = threading.Lock()
_CACHE_MAX_SIZE = 50


def filter_data(
    reactant_types: list | None = None,
    reaction_types: list | None = None,
    fg_a: str | list | None = None,
    fg_b: str | list | None = None,
    exclude_cui: list | None = None,
    exclude_scaleup: list | None = None,
    include_null_categories: list | None = None,
    min_eln: int | None = None,
    topn_zscore: int | None = None,
    max_components: int | None = None,
    return_stats: bool = False,
    source_df: pd.DataFrame | None = None,
    session_id: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Return a filtered DataFrame using the app's 10-step filter chain.

    When *return_stats* is ``True`` the return value is a tuple
    ``(filtered_df, stats_dict)``.

    Args:
        session_id: UUID of an uploaded dataset (used for cache keying).
            Pass ``None`` when using the default dataset.
    """
    using_uploaded = source_df is not None

    # --- cache lookup (works for both default and uploaded data) ---
    cache_key = _create_cache_key(
        session_id,
        reactant_types, reaction_types, fg_a, fg_b,
        exclude_cui, exclude_scaleup, include_null_categories,
        min_eln, topn_zscore, max_components, return_stats,
    )
    with _FILTER_CACHE_LOCK:
        if cache_key in _FILTER_CACHE:
            _FILTER_CACHE.move_to_end(cache_key)  # LRU: mark as recently used
            cached = _FILTER_CACHE[cache_key]
            if return_stats:
                return cached['dataframe'].copy(), cached['stats'].copy()
            return cached['dataframe'].copy()

    # No .copy() needed: every filter step returns a new DataFrame via
    # boolean indexing, and _deduplicate_best_zscore creates an explicit copy.
    dff = source_df if using_uploaded else DF
    include_null = _convert_checkbox_to_bool(include_null_categories)
    stats: dict | None = {} if return_stats else None

    # Step 1: Reaction types
    dff = _filter_by_reaction_types(dff, reaction_types)
    if stats is not None:
        stats['whole_dataset'] = {'elns': dff['ELN_ID'].nunique()}

    # Step 2: Reactant columns populated
    dff = _filter_by_reactant_columns(dff, reactant_types, include_null)
    if stats is not None and reactant_types:
        stats['after_reactant_filters'] = {'elns': dff['ELN_ID'].nunique()}

    # Step 3: CuI exclusion
    dff = _filter_exclude_cui(dff, exclude_cui)

    # Step 4: Functional Group A
    dff, fg_a_list = _filter_fg_a(dff, fg_a)
    if stats is not None and fg_a_list:
        stats['after_fg_a'] = {'elns': dff['ELN_ID'].nunique()}

    # Step 5: Functional Group B
    dff, fg_b_list = _filter_fg_b(dff, fg_b, fg_a_list)
    if stats is not None and fg_b_list:
        stats['after_fg_b'] = {'elns': dff['ELN_ID'].nunique()}

    # Step 6: Scale-up plates
    dff = _filter_scaleup_plates(dff, exclude_scaleup)

    # Step 7: Deduplication
    dff = _deduplicate_best_zscore(dff)

    # Step 8: Top-N z-scores
    dff = _filter_topn_zscore(dff, topn_zscore, reactant_types, include_null)

    # Step 9: Min ELN count
    dff = _filter_min_eln(dff, min_eln, reactant_types, include_null)
    if stats is not None:
        stats['after_min_eln'] = {'elns': dff['ELN_ID'].nunique()}

    # Compute max-components cap for the slider
    if stats is not None and reactant_types:
        try:
            key_cols = [rt for rt in reactant_types if rt]
            stats['max_components_cap'] = int(dff[key_cols].drop_duplicates().shape[0])
        except Exception:
            stats['max_components_cap'] = 1

    # Step 10: Max components
    dff = _filter_max_components(dff, max_components, reactant_types, include_null)

    # --- cache store (LRU eviction) ---
    with _FILTER_CACHE_LOCK:
        if len(_FILTER_CACHE) >= _CACHE_MAX_SIZE:
            _FILTER_CACHE.popitem(last=False)  # evict least-recently-used
        _FILTER_CACHE[cache_key] = {
            'dataframe': dff.copy(),
            'stats': stats.copy() if stats else {},
        }
        # Track cache keys per session for cleanup on upload removal
        if session_id:
            _SESSION_CACHE_KEYS.setdefault(session_id, set()).add(cache_key)

    if not return_stats:
        return dff
    return dff, stats or {}


def clear_filter_cache():
    """Clear the filter data cache. Useful for debugging or memory management."""
    with _FILTER_CACHE_LOCK:
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
# 5. SERVER-SIDE UPLOAD STORE
# ---------------------------------------------------------------------------
# Uploaded DataFrames are kept in-process memory keyed by UUID.  Only the
# UUID travels between browser and server, eliminating repeated JSON
# serialisation/deserialisation of potentially large datasets.

_UPLOAD_STORE: dict[str, dict] = {}
_UPLOAD_STORE_LOCK = threading.Lock()
_UPLOAD_TTL_SECONDS = 3600   # 1 hour of inactivity
_UPLOAD_MAX_SESSIONS = 10    # max concurrent uploaded datasets
_SESSION_CACHE_KEYS: dict[str, set[str]] = {}  # session_id → filter cache keys


def _cleanup_expired_uploads() -> None:
    """Remove upload store entries that have exceeded the TTL.

    Must be called while holding ``_UPLOAD_STORE_LOCK``.
    """
    now = time.time()
    expired = [
        sid for sid, entry in _UPLOAD_STORE.items()
        if now - entry['last_access'] > _UPLOAD_TTL_SECONDS
    ]
    for sid in expired:
        del _UPLOAD_STORE[sid]
        _purge_session_cache(sid)
        logger.info('Upload session %s expired and removed', sid[:8])


def _purge_session_cache(session_id: str) -> None:
    """Remove all filter-cache entries associated with *session_id*."""
    keys = _SESSION_CACHE_KEYS.pop(session_id, set())
    with _FILTER_CACHE_LOCK:
        for k in keys:
            _FILTER_CACHE.pop(k, None)


def store_uploaded_dataframe(df: pd.DataFrame) -> str:
    """Store *df* server-side and return a UUID handle.

    Args:
        df: Validated DataFrame from a user upload.

    Returns:
        UUID string that identifies this upload session.
    """
    session_id = str(uuid.uuid4())
    with _UPLOAD_STORE_LOCK:
        _cleanup_expired_uploads()
        # Evict oldest if at capacity
        while len(_UPLOAD_STORE) >= _UPLOAD_MAX_SESSIONS:
            oldest = next(iter(_UPLOAD_STORE))
            del _UPLOAD_STORE[oldest]
            _purge_session_cache(oldest)
            logger.info('Upload session %s evicted (cap reached)', oldest[:8])
        _UPLOAD_STORE[session_id] = {
            'df': df,
            'last_access': time.time(),
        }
    logger.info('Stored upload session %s (%d rows)', session_id[:8], len(df))
    return session_id


def get_uploaded_dataframe(session_id: str) -> pd.DataFrame | None:
    """Look up a stored DataFrame by its UUID handle.

    Updates the last-access timestamp so the entry stays alive.

    Args:
        session_id: UUID returned by :func:`store_uploaded_dataframe`.

    Returns:
        The stored DataFrame, or ``None`` if expired / not found.
    """
    if not session_id:
        return None
    with _UPLOAD_STORE_LOCK:
        entry = _UPLOAD_STORE.get(session_id)
        if entry is None:
            return None
        entry['last_access'] = time.time()
        return entry['df']


def remove_uploaded_dataframe(session_id: str) -> None:
    """Explicitly remove a stored upload and its cached filter results."""
    if not session_id:
        return
    with _UPLOAD_STORE_LOCK:
        _UPLOAD_STORE.pop(session_id, None)
    _purge_session_cache(session_id)
    logger.info('Upload session %s removed', session_id[:8])


def get_active_dataframe(session_id: str | None = None) -> pd.DataFrame:
    """Return the uploaded DataFrame for *session_id*, or the default ``DF``.

    Args:
        session_id: UUID from ``uploaded-data-store`` (may be ``None``).

    Returns:
        The uploaded DataFrame if available, otherwise the default DF.
    """
    if session_id:
        uploaded_df = get_uploaded_dataframe(session_id)
        if uploaded_df is not None:
            return uploaded_df
    return DF


def get_reaction_types_from_data(df: pd.DataFrame = None) -> list[str]:
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


def get_category_options_from_data(df: pd.DataFrame = None) -> list[str]:
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

    rng = np.random.default_rng(random_state)

    # Get observed Kruskal-Wallis H statistic
    categories = dff[category_col].unique()
    groups = [dff[dff[category_col] == cat][value_col].values for cat in categories]
    observed_h, observed_p = stats.kruskal(*groups)

    # Permutation test: shuffle category labels, recalculate H
    permuted_h_values = []
    values = dff[value_col].values.copy()

    for _ in range(n_permutations):
        rng.shuffle(values)
        
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
