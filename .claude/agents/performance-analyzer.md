---
name: performance-analyzer
description: Finds performance bottlenecks in data processing and visualization code
tools:
  - Read
  - Glob
  - Grep
  - Bash
---

# Performance Analyzer Agent

Analyze the codebase for performance issues:

## Data Processing
- Check `filterChain.ts` filter steps for unnecessary array copies or redundant operations
- Verify filter chain runs in <50ms for 67K rows
- Look for O(n^2) patterns in filter steps or dropdown option computation
- Check `dropdownOptions.ts` for expensive conditioning logic

## Rendering
- Check Plotly configs in `plots/*.ts` for traces with excessive data points
- Verify adaptive height calculations don't cause layout thrashing
- Look for unnecessary React re-renders (missing useMemo, unstable props)
- Check that `Plot.tsx` wrapper doesn't re-render on every state change

## Memory
- Look for data array copies that aren't necessary
- Check Zustand store for stale state that should be cleaned up
- Verify uploaded dataset storage doesn't leak memory
- Check for closures retaining large data arrays

## Bundle Size
- Verify plotly.js-dist-min is used (not full plotly.js)
- Look for unnecessary imports that increase bundle size
- Check for tree-shaking opportunities

## Startup
- Check Parquet loading and parsing performance
- Verify version manifest fetching doesn't block initial render
- Look for unnecessary work during initial mount

Output findings as a prioritized list with estimated impact (high/medium/low).
