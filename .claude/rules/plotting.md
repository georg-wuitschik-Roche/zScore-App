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
tooltip.ts  → TooltipSource payload + buildTooltipModel() for the custom hover tooltip
```

## Generic Builder Pattern
`helpers.ts` contains `buildDistributionConfig(rows, reactantTypes, presentationMode, buildTrace, ...)`. This groups data, sorts by median, and calls `buildTrace(group)` for each category. Boxplot and violin plug in different trace builders but share all the data preparation, layout, colorbar, and rank annotation logic.

When adding a new distribution plot type, implement a trace builder function and pass it to `buildDistributionConfig` — don't duplicate the grouping/sorting/layout logic.

## Color Mapping
Use `colors.ts` for ELN density-based color interpolation. Never hardcode colors in plot configs. The density is based on unique ELN count per category.

## Hover Tooltips
Plotly's native SVG hover label is **disabled** on all data traces — it is non-interactive
and vanishes on unhover, so values could not be selected or copied. A custom HTML tooltip
(`components/PlotTooltip.tsx` + `hooks/usePlotTooltip.ts`) renders instead, and stays open
while the cursor is over it.

`buildDistributionConfig()` applies the hover contract itself, so boxplot/violin trace
builders return **geometry only** — never set `hoverinfo`, `customdata` or `hovertemplate`
there. Standalone builders (heatmap) apply it directly:

```ts
// 'none', NOT 'skip'. 'skip' removes the trace from the hover search entirely, so
// plotly_hover never fires. 'none' fires the event and suppresses the label.
hoverinfo: 'none' as const,
customdata: asCustomdata(cells),   // TooltipSource[] — [][] for heatmap
// Never set hovertemplate: a set hovertemplate renders a label regardless of hoverinfo.
```

`customdata` carries a `TooltipSource` per point (`plots/tooltip.ts`), not display strings.
`buildTooltipModel()` expands it into label/value rows lazily on hover — don't build the
display model at prepare time, that runs in the filter-hot path. **Row values must be plain
text**: each is copied verbatim to the clipboard on click, so no HTML markup.

`tooltip.ts` owns both ends of the Plotly type override: `asCustomdata()` going in,
`readTooltipHover()` coming back out of a hover event. Don't cast `customdata` elsewhere.

For distribution traces `customdata` must stay index-aligned with `zScores`, because Plotly
maps a hovered box/violin point back to its *original input index*.

Bind Plotly events with `bindPlotlyEvent()` from `components/Plot.tsx`, called from
`onInitialized` — react-plotly.js's `onHover`/`onUnhover`/`onRelayout` props do not work
here. `usePlotChrome()` already wires zoom tracking and the tooltip for both plot views.

Rank badge annotations are the one exception — they keep Plotly's native label via the
module-private `getHoverLabelStyle()` in `helpers.ts`.

## Dark Mode
All plot configs receive an `isDark` boolean. Use it for:
- Grid/axis colors, text colors, background
- Colorbar styling
- Rank badge colors (`rankBadgeColor()`) and their native hover label

The hover tooltip is exempt — it is HTML and picks up dark mode from CSS custom properties.

## Adaptive Height
`max(800, numCategories * 110)` — computed in `prepareDistributionData()`. Presentation mode adds larger fonts (title, axes).

## Comparison Rank Annotations
When comparison mode is active, `buildRankAnnotation()` adds badges next to each y-axis category showing rank change (NEW, up, down, unchanged). These are Plotly annotation objects with hover text showing detailed comparison info.

## Export
PNG export via `Plotly.toImage()` at high resolution. Strip unnecessary annotations for clean exports.
