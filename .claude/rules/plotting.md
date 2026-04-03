---
paths:
  - "frontend/src/plots/**"
---

# Plotting Conventions (Plotly)

## Architecture
```
helpers.ts (core)
  ├── prepareDistributionData()   — groups rows, sorts by median, builds customdata
  ├── buildDistributionConfig()   — generic builder: takes a per-group trace builder fn
  ├── buildMedianTrace()          — invisible scatter for median tooltip
  └── buildRankAnnotation()       — comparison rank badge

boxplot.ts  → calls buildDistributionConfig() with box trace builder
violin.ts   → calls prepareDistributionData() + adds KDE-bounded median lines
heatmap.ts  → standalone: builds z-matrix from rows (no shared builder)
colors.ts   → ELN density → interpolated color
```

## Generic Builder Pattern
`helpers.ts` contains `buildDistributionConfig(rows, reactantTypes, presentationMode, buildTrace, ...)`. This groups data, sorts by median, and calls `buildTrace(group)` for each category. Boxplot and violin plug in different trace builders but share all the data preparation, layout, colorbar, and rank annotation logic.

When adding a new distribution plot type, implement a trace builder function and pass it to `buildDistributionConfig` — don't duplicate the grouping/sorting/layout logic.

## Color Mapping
Use `colors.ts` for ELN density-based color interpolation. Never hardcode colors in plot configs. The density is based on unique ELN count per category.

## Hover Templates
Use `customdata` + `hovertemplate` pattern for rich hover info. Never use `text` + `hoverinfo`.

```ts
// Good — customdata array built per-row in prepareDistributionData
hovertemplate: "<b>%{customdata[0]}</b><br>z-Score: %{x:.2f}<extra></extra>"
```

## Dark Mode
All plot configs receive an `isDark` boolean. Use it for:
- Grid/axis colors, text colors, background
- Colorbar styling
- Hover label backgrounds (`getHoverLabelStyle(isDark)`)
- Rank badge colors (`rankBadgeColor()`)

## Adaptive Height
`max(800, numCategories * 110)` — computed in `prepareDistributionData()`. Presentation mode adds larger fonts (title, axes).

## Comparison Rank Annotations
When comparison mode is active, `buildRankAnnotation()` adds badges next to each y-axis category showing rank change (NEW, up, down, unchanged). These are Plotly annotation objects with hover text showing detailed comparison info.

## Export
PNG export via `Plotly.toImage()` at high resolution. Strip unnecessary annotations for clean exports.
