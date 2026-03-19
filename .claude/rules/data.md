---
paths:
  - "data_utils.py"
---

# Data Processing Conventions

## Global DataFrame
The main dataset is loaded once at import time as `DF`. Use copy-on-write semantics — never mutate the global.

```python
# Good - work on a copy
dff = DF.copy()
dff = dff[dff["Reaction Type"].isin(selected)]

# Bad - mutating global
DF.drop(columns=["unused"], inplace=True)
```

## Filter Chain
All filtering goes through `filter_data()`. Filters apply in a specific order for performance (cheap filters first). Never add ad-hoc filtering in callbacks.

```python
# Good - use the central filter function
filtered_df, stats = filter_data(
    df=df,
    reaction_types=reaction_types,
    reactant_type=reactant_type,
    min_elns=min_elns,
    top_n=top_n,
)

# Bad - filtering in callbacks
dff = df[df["Base"] == "Et3N"]
dff = dff[dff["z-Score"] > 0]
```

## Caching
Use the existing LRU-style cache with MD5 hash keys. Skip cache for uploaded data.

```python
# Good
cache_key = hashlib.md5(
    str(sorted(params.items())).encode()
).hexdigest()

if cache_key in _filter_cache:
    return _filter_cache[cache_key]
```

## Column Access
Use string column names, not positional indexing. Validate required columns exist before processing.

```python
# Good
df["z-Score"].median()
required = {"ELN_ID", "z-Score", "Reaction Type"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Bad
df.iloc[:, 12]
```

## Decimal Handling
European CSV files use comma as decimal separator. Always handle this in parsing.

```python
# Good
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
```
