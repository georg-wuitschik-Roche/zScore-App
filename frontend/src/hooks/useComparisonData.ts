/**
 * Hook that computes rank deltas between the current and comparison dataset versions.
 *
 * Returns null when comparison mode is off or comparison data is unavailable.
 * Runs the same filter chain on the comparison dataset so ranks are comparable.
 */

import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { filterData } from '../data/filterChain';
import { computeRankDeltas, resolveComparisonVersion } from '../data/comparison';
import type { Row, RankDelta, FilterParams } from '../data/types';

export interface ComparisonResult {
  /** Rank deltas keyed by compound reactant key (e.g. "PdCl2 / DMF") */
  rankMap: Map<string, RankDelta>;
  /** Per-axis rank deltas for heatmaps: one map per individual reactant type column */
  axisRankMaps: Map<string, RankDelta>[];
}

/**
 * Compute rank deltas for the current filtered rows vs the comparison version.
 *
 * @param currentRows - Already-filtered rows from the active version
 * @param reactantTypes - Active reactant types (used for grouping)
 */
export function useComparisonRanks(
  currentRows: Row[],
  reactantTypes: string[],
): ComparisonResult | null {
  const comparisonMode = useFilterStore((s) => s.comparisonMode);
  const comparisonVersion = useFilterStore((s) => s.comparisonVersion);
  const activeVersion = useFilterStore((s) => s.activeVersion);
  const availableVersions = useFilterStore((s) => s.availableVersions);
  const datasetCache = useFilterStore((s) => s.datasetCache);

  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const fgA = useFilterStore((s) => s.fgA);
  const fgB = useFilterStore((s) => s.fgB);
  const excludeCui = useFilterStore((s) => s.excludeCui);
  const excludeScaleup = useFilterStore((s) => s.excludeScaleup);
  const includeNullCategories = useFilterStore((s) => s.includeNullCategories);
  const minEln = useFilterStore((s) => s.minEln);
  const topnZscore = useFilterStore((s) => s.topnZscore);
  const maxComponents = useFilterStore((s) => s.maxComponents);

  // Determine which version to compare against
  const comparisonVersionId = useMemo(() => {
    if (!comparisonMode) return null;
    return resolveComparisonVersion(availableVersions, activeVersion, comparisonVersion);
  }, [comparisonMode, comparisonVersion, activeVersion, availableVersions]);

  // Get comparison dataset rows from cache
  const comparisonRows = useMemo(() => {
    if (!comparisonVersionId) return null;
    return datasetCache[comparisonVersionId]?.rows ?? null;
  }, [comparisonVersionId, datasetCache]);

  return useMemo(() => {
    if (!comparisonMode || !comparisonRows || reactantTypes.length === 0) return null;

    const params: FilterParams = {
      reactionTypes,
      reactantTypes,
      fgA,
      fgB,
      excludeCui,
      excludeScaleup,
      includeNullCategories,
      minEln,
      topnZscore,
      maxComponents,
    };

    const comparisonFiltered = filterData(comparisonRows, params);

    // Compound-key rank deltas (for boxplot/violin/stats)
    const rankMap = computeRankDeltas(currentRows, comparisonFiltered.rows, reactantTypes);

    // Per-axis rank deltas (for heatmap axes)
    const axisRankMaps = reactantTypes.map((col) =>
      computeRankDeltas(currentRows, comparisonFiltered.rows, [col]),
    );

    return { rankMap, axisRankMaps };
  }, [
    comparisonMode,
    comparisonRows,
    currentRows,
    reactantTypes,
    reactionTypes,
    fgA,
    fgB,
    excludeCui,
    excludeScaleup,
    includeNullCategories,
    minEln,
    topnZscore,
    maxComponents,
  ]);
}
