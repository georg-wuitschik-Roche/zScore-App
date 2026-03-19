---
paths:
  - "*.py"
  - "**/*.py"
---

# Python Conventions

## Type Hints
Every module MUST start with `from __future__ import annotations`. All functions require type hints on parameters and return values.

```python
# Good
from __future__ import annotations

def filter_data(
    df: pd.DataFrame,
    reaction_types: list[str],
    min_elns: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Filter dataset by reaction types.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    reaction_types : list[str]
        Reaction types to include.
    min_elns : int
        Minimum ELN count threshold.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Filtered dataframe and stats dict.
    """

# Bad - no future annotations, no type hints, no docstring
def filter_data(df, reaction_types, min_elns=5):
    pass
```

## Naming
- Private functions: `_prefix` (e.g. `_load_and_prepare`)
- Constants: `UPPER_SNAKE_CASE` at module top
- Variables/functions: `snake_case`
- No abbreviations except well-known ones (df, fig, idx)

```python
# Good
CACHE_MAX_SIZE = 50
BASE_COLOURS = {"Base": ("#cce5ff", "#004085")}

def _compute_cache_key(params: dict) -> str: ...

# Bad
cacheMax = 50
def compKey(p): ...
```

## Imports
Order: stdlib, third-party, local. One blank line between groups.

```python
# Good
from __future__ import annotations
import hashlib
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from data_utils import filter_data, DF
```

## Error Handling
Use specific exceptions. Log errors for debugging but provide user-friendly messages for UI callbacks.

```python
# Good
try:
    df = pd.read_csv(path, encoding=encoding)
except UnicodeDecodeError:
    df = pd.read_csv(path, encoding="latin-1")
except FileNotFoundError:
    raise ValueError(f"Data file not found: {path}")
```

## No `any` Equivalent
Avoid bare `dict` or `list` without type parameters. Use specific types.

```python
# Good
stats: dict[str, int | float] = {}
options: list[dict[str, str]] = []

# Bad
stats: dict = {}
options: list = []
```
