---
paths:
  - "plot_utils.py"
  - "export_boxplots.py"
  - "generate_supplementary_figures.py"
---

# Plotting Conventions (Plotly)

## Function Signatures
Plot functions return `tuple[go.Figure, int]` — the figure and computed height.

```python
# Good
def create_boxplot(
    dff: pd.DataFrame,
    reactant_types: list[str],
    base_height: int = 800,
    presentation_mode: bool = False,
    reaction_type: str | None = None,
    max_categories: int | None = None,
) -> tuple[go.Figure, int]:
    ...
```

## Color Mapping
Use `BASE_COLOURS` dict with `(light, dark)` tuples. Interpolate based on ELN count density.

```python
# Good - use existing color map
color = BASE_COLOURS.get(category, ("#e0e0e0", "#333333"))
interpolated = interpolate_color(color[0], color[1], density)

# Bad - hardcoded hex in plot calls
fig.add_trace(go.Box(marker_color="#ff6347"))
```

## Hover Templates
Use comprehensive HTML hover templates showing all relevant experiment details.

```python
# Good - rich hover info
hovertemplate = (
    "<b>%{customdata[0]}</b><br>"
    "Plate: %{customdata[1]} | Coord: %{customdata[2]}<br>"
    "z-Score: %{x:.2f}<br>"
    "<extra></extra>"
)
```

## Export Settings
- Publication plots: 48px title, 32px body text
- PNG: 1600px wide, 4x scale factor
- Strip titles for paper exports
- Safe filenames: spaces → underscores, slashes → hyphens

```python
# Good
fig.write_image(
    path,
    width=1600,
    height=computed_height,
    scale=4,
    format="png",
)
```

## Adaptive Height
Calculate height dynamically based on category count.

```python
# Good
height = max(base_height, len(categories) * 110 + overhead)
```
