---
paths:
  - "frontend/src/components/**"
  - "frontend/src/App.tsx"
---

# React Component Conventions

## Structure
- One component per file, named export matching filename
- Function components with TypeScript props interfaces
- No class components

## State
- All shared state lives in `filterStore.ts` via Zustand
- Use named selectors from the store, never `useStore()` with full state
- Local UI state (hover, open/closed) uses `useState`
- Derived data uses `useMemo` — never compute in render body without memoization

```tsx
// Good - named selector
const reactionTypes = useFilterStore(s => s.reactionTypes)

// Bad - selecting entire store
const store = useFilterStore()
```

## Key Component Patterns

### DistributionView — Generic Plot Renderer
`DistributionView` renders both boxplot and violin tabs. It takes a `buildConfig` function prop that creates the PlotConfig. AnalysisTabs passes either `createBoxplotConfig` or `createViolinConfig`. Don't create separate view components for new distribution plot types — add a new config builder in `plots/` and pass it to DistributionView.

### AnalysisTabs — Tab Switcher + Split Grid
- Tabs: boxplot, violin, heatmap, stats (TabId type in `types.ts`)
- Heatmap requires 2+ reactant types — automatically hidden otherwise
- If current tab becomes unavailable, falls back to boxplot
- Uses `useDeferredValue()` on split panels for responsive UI during heavy computation
- Comparison data is fetched once, not per panel

### Split Mode Rendering
When split mode is active (2+ values on a filter), `useSplitFilteredData` returns multiple `SplitPanel` objects. AnalysisTabs renders them in a CSS grid. Each panel gets its own filtered data, rank map, and comparison info.

## Styling
- All styles in `styles/app.css` using CSS custom properties
- Only use inline styles for truly dynamic values (computed heights, etc.)
- Design tokens defined in `:root` — never hardcode colors or fonts
- Dark mode: store tracks `theme` (resolved) and `themePreference` ('auto'|'light'|'dark')

## Plotly Integration
- Use the `Plot` wrapper component (dist-min bundle) — never import plotly directly
- Plot configs are built in `plots/*.ts`, not inline in components
- Use `useCallback` for Plotly event handlers
