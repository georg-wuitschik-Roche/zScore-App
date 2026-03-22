"""Shared test fixtures for the zScore-App test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Synthetic small dataset (~200 rows)
# ---------------------------------------------------------------------------

_REACTION_TYPES = ['Buchwald-Hartwig', 'Suzuki-Miyaura', 'C-H Activation']
_ELN_IDS = [f'ELN-{rt[:2].upper()}-{i:02d}' for rt in _REACTION_TYPES for i in range(1, 6)]
_CATALYSTS = ['Pd(OAc)2', 'Pd2(dba)3', 'CuI', 'NiCl2', 'Pd(PPh3)4']
_SOLVENTS = ['DMF', 'DMSO', 'THF', 'Toluene', 'DCM']
_BASES = ['Cs2CO3', 'K2CO3', 'K3PO4', 'Et3N', 'DBU']
_LIGANDS = ['XPhos', 'SPhos', 'PPh3', 'BINAP', None]
_FG_A_VALUES = ['RNH2', 'RNH2 a-branch', 'ArNH2', 'ArOH', 'RCOOH']
_FG_B_VALUES = ['ArBr', 'ArCl', 'ArI', 'ArOTf', 'ArBpin']
_PLATES = [1, 2, 3]


def _build_small_df() -> pd.DataFrame:
    """Build a synthetic ~200-row DataFrame covering all filter edge cases."""
    rng = np.random.RandomState(42)
    rows = []
    row_id = 0

    for _rt_idx, rt in enumerate(_REACTION_TYPES):
        eln_ids = [eid for eid in _ELN_IDS if eid.startswith(f'ELN-{rt[:2].upper()}')]
        for eln_idx, eln_id in enumerate(eln_ids):
            n_rows = 10 + (eln_idx % 3) * 2  # 10, 12, or 14 rows per ELN
            for j in range(n_rows):
                # Deterministic z-score: based on catalyst rank so median ordering is stable
                cat_idx = (eln_idx + j) % len(_CATALYSTS)
                catalyst = _CATALYSTS[cat_idx]
                base_zscore = 3.0 - cat_idx * 0.5  # Pd(OAc)2 highest, Pd(PPh3)4 lowest
                z_score = base_zscore + rng.normal(0, 0.3)

                fg_a = _FG_A_VALUES[(eln_idx + j) % len(_FG_A_VALUES)]
                fg_b = _FG_B_VALUES[(eln_idx + j) % len(_FG_B_VALUES)]
                fg_sorted = ', '.join(sorted([fg_a, fg_b]))

                ligand = _LIGANDS[(eln_idx + j) % len(_LIGANDS)]
                plate = _PLATES[j % len(_PLATES)]

                rows.append(
                    {
                        'ELN_ID': eln_id,
                        'PLATENUMBER': plate,
                        'Coordinate': f'{chr(65 + j % 8)}{(j % 12) + 1}',
                        'AREA_TOTAL_REDUCED': 100.0 + rng.normal(0, 20),
                        'Reaction Type': rt,
                        'Catalyst': catalyst,
                        'Solvent': _SOLVENTS[(eln_idx + j) % len(_SOLVENTS)],
                        'Base': _BASES[(eln_idx + j) % len(_BASES)],
                        'Ligand': ligand,
                        'Additive': None,
                        'Coupling Reagent': None,
                        'Secondary Solvent': None,
                        'FG A': fg_a,
                        'FG B': fg_b,
                        'FG_sorted': fg_sorted,
                        'FG_PAIR_SORTED': fg_sorted,
                        'z-Score': z_score,
                    }
                )
                row_id += 1

    # Add a few special rows for edge cases:
    # 1. Scale-up plate (no reagent variability) — same ELN, same plate, same reagents
    for j in range(5):
        rows.append(
            {
                'ELN_ID': 'ELN-SCALEUP',
                'PLATENUMBER': 99,
                'Coordinate': f'A{j + 1}',
                'AREA_TOTAL_REDUCED': 150.0,
                'Reaction Type': 'Buchwald-Hartwig',
                'Catalyst': 'Pd(OAc)2',
                'Solvent': 'DMF',
                'Base': 'Cs2CO3',
                'Ligand': 'XPhos',
                'Additive': None,
                'Coupling Reagent': None,
                'Secondary Solvent': None,
                'FG A': 'RNH2',
                'FG B': 'ArBr',
                'FG_sorted': 'ArBr, RNH2',
                'FG_PAIR_SORTED': 'ArBr, RNH2',
                'z-Score': 2.0 + j * 0.1,
            }
        )

    # 2. Rows with null/missing values in reactant columns
    for j in range(3):
        rows.append(
            {
                'ELN_ID': 'ELN-NULL',
                'PLATENUMBER': 1,
                'Coordinate': f'B{j + 1}',
                'AREA_TOTAL_REDUCED': 80.0,
                'Reaction Type': 'Buchwald-Hartwig',
                'Catalyst': None,
                'Solvent': 'DMF',
                'Base': None,
                'Ligand': None,
                'Additive': None,
                'Coupling Reagent': None,
                'Secondary Solvent': None,
                'FG A': 'RNH2',
                'FG B': 'ArBr',
                'FG_sorted': 'ArBr, RNH2',
                'FG_PAIR_SORTED': 'ArBr, RNH2',
                'z-Score': 0.5 + j * 0.1,
            }
        )

    df = pd.DataFrame(rows)

    # Convert to categorical to match real data
    cat_cols = [
        'Catalyst',
        'Solvent',
        'Base',
        'Ligand',
        'Additive',
        'Coupling Reagent',
        'Secondary Solvent',
        'Reaction Type',
        'FG A',
        'FG B',
        'FG_PAIR_SORTED',
        'ELN_ID',
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df


def _build_minimal_df() -> pd.DataFrame:
    """Build a ~20-row DataFrame for testing individual filter steps."""
    rows = [
        # Normal rows (Buchwald-Hartwig)
        {
            'ELN_ID': 'E1',
            'PLATENUMBER': 1,
            'Coordinate': 'A1',
            'AREA_TOTAL_REDUCED': 100.0,
            'Reaction Type': 'Buchwald-Hartwig',
            'Catalyst': 'Pd(OAc)2',
            'Solvent': 'DMF',
            'Base': 'Cs2CO3',
            'Ligand': 'XPhos',
            'FG A': 'RNH2',
            'FG B': 'ArBr',
            'FG_sorted': 'ArBr, RNH2',
            'FG_PAIR_SORTED': 'ArBr, RNH2',
            'z-Score': 3.0,
        },
        {
            'ELN_ID': 'E1',
            'PLATENUMBER': 1,
            'Coordinate': 'A2',
            'AREA_TOTAL_REDUCED': 110.0,
            'Reaction Type': 'Buchwald-Hartwig',
            'Catalyst': 'Pd(OAc)2',
            'Solvent': 'DMF',
            'Base': 'Cs2CO3',
            'Ligand': 'XPhos',
            'FG A': 'RNH2',
            'FG B': 'ArBr',
            'FG_sorted': 'ArBr, RNH2',
            'FG_PAIR_SORTED': 'ArBr, RNH2',
            'z-Score': 2.5,
        },
        # CuI catalyst row
        {
            'ELN_ID': 'E2',
            'PLATENUMBER': 1,
            'Coordinate': 'B1',
            'AREA_TOTAL_REDUCED': 90.0,
            'Reaction Type': 'Buchwald-Hartwig',
            'Catalyst': 'CuI',
            'Solvent': 'DMSO',
            'Base': 'K2CO3',
            'Ligand': 'PPh3',
            'FG A': 'ArNH2',
            'FG B': 'ArCl',
            'FG_sorted': 'ArCl, ArNH2',
            'FG_PAIR_SORTED': 'ArCl, ArNH2',
            'z-Score': 1.5,
        },
        # Different reaction type
        {
            'ELN_ID': 'E3',
            'PLATENUMBER': 1,
            'Coordinate': 'C1',
            'AREA_TOTAL_REDUCED': 120.0,
            'Reaction Type': 'Suzuki-Miyaura',
            'Catalyst': 'Pd(PPh3)4',
            'Solvent': 'THF',
            'Base': 'K3PO4',
            'Ligand': 'SPhos',
            'FG A': 'ArOH',
            'FG B': 'ArBpin',
            'FG_sorted': 'ArBpin, ArOH',
            'FG_PAIR_SORTED': 'ArBpin, ArOH',
            'z-Score': 2.0,
        },
        # Null catalyst row
        {
            'ELN_ID': 'E4',
            'PLATENUMBER': 1,
            'Coordinate': 'D1',
            'AREA_TOTAL_REDUCED': 80.0,
            'Reaction Type': 'Buchwald-Hartwig',
            'Catalyst': None,
            'Solvent': 'DMF',
            'Base': None,
            'Ligand': None,
            'FG A': 'RNH2',
            'FG B': 'ArBr',
            'FG_sorted': 'ArBr, RNH2',
            'FG_PAIR_SORTED': 'ArBr, RNH2',
            'z-Score': 0.5,
        },
    ]
    # Add more rows to reach ~20
    for i in range(15):
        rows.append(
            {
                'ELN_ID': f'E{5 + i // 3}',
                'PLATENUMBER': 1 + (i % 2),
                'Coordinate': f'E{i + 1}',
                'AREA_TOTAL_REDUCED': 100.0 + i,
                'Reaction Type': 'Buchwald-Hartwig',
                'Catalyst': _CATALYSTS[i % len(_CATALYSTS)],
                'Solvent': _SOLVENTS[i % len(_SOLVENTS)],
                'Base': _BASES[i % len(_BASES)],
                'Ligand': _LIGANDS[i % len(_LIGANDS)],
                'FG A': _FG_A_VALUES[i % len(_FG_A_VALUES)],
                'FG B': _FG_B_VALUES[i % len(_FG_B_VALUES)],
                'FG_sorted': ', '.join(
                    sorted(
                        [
                            _FG_A_VALUES[i % len(_FG_A_VALUES)],
                            _FG_B_VALUES[i % len(_FG_B_VALUES)],
                        ]
                    )
                ),
                'FG_PAIR_SORTED': ', '.join(
                    sorted(
                        [
                            _FG_A_VALUES[i % len(_FG_A_VALUES)],
                            _FG_B_VALUES[i % len(_FG_B_VALUES)],
                        ]
                    )
                ),
                'z-Score': 1.0 + (i % 5) * 0.5,
            }
        )

    df = pd.DataFrame(rows)
    for col in ['Catalyst', 'Solvent', 'Base', 'Ligand', 'Reaction Type', 'FG A', 'FG B', 'FG_PAIR_SORTED', 'ELN_ID']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Synthetic ~200 row dataset with 3 reaction types and known values."""
    return _build_small_df()


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal ~20 row dataset for testing individual filter steps."""
    return _build_minimal_df()


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """DataFrame with correct columns but zero rows."""
    cols = [
        'ELN_ID',
        'PLATENUMBER',
        'Coordinate',
        'AREA_TOTAL_REDUCED',
        'Reaction Type',
        'Catalyst',
        'Solvent',
        'Base',
        'Ligand',
        'Additive',
        'Coupling Reagent',
        'Secondary Solvent',
        'FG A',
        'FG B',
        'FG_sorted',
        'FG_PAIR_SORTED',
        'z-Score',
    ]
    return pd.DataFrame(columns=cols)


@pytest.fixture
def upload_df() -> pd.DataFrame:
    """Valid DataFrame matching REQUIRED_COLUMNS for upload tests."""
    n = 50
    rng = np.random.RandomState(99)
    return pd.DataFrame(
        {
            'ELN_ID': [f'UPL-{i:03d}' for i in range(n)],
            'PLATENUMBER': rng.randint(1, 4, n),
            'Coordinate': [f'A{i % 12 + 1}' for i in range(n)],
            'AREA_TOTAL_REDUCED': rng.normal(100, 20, n),
            'Base': rng.choice(['Cs2CO3', 'K2CO3'], n),
            'Catalyst': rng.choice(['Pd(OAc)2', 'CuI'], n),
            'Solvent': rng.choice(['DMF', 'DMSO'], n),
            'Ligand': rng.choice(['XPhos', 'SPhos', None], n),
            'Reaction Type': rng.choice(['Buchwald-Hartwig', 'Suzuki-Miyaura'], n),
            'FG A': rng.choice(['RNH2', 'ArNH2'], n),
            'FG B': rng.choice(['ArBr', 'ArCl'], n),
            'FG_sorted': 'ArBr, RNH2',
            'z-Score': rng.normal(0, 1.5, n),
        }
    )


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear filter cache and upload store before and after each test."""
    import data_utils as du

    du.clear_filter_cache()
    du._UPLOAD_STORE.clear()
    du._SESSION_CACHE_KEYS.clear()
    yield
    du.clear_filter_cache()
    du._UPLOAD_STORE.clear()
    du._SESSION_CACHE_KEYS.clear()


@pytest.fixture(scope='session')
def full_dataset() -> pd.DataFrame:
    """Load the real dataset once per test session. Only for @pytest.mark.slow tests."""
    import data_utils as du

    return du.DF.copy()
