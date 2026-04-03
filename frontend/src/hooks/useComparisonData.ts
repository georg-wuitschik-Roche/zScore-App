/**
 * Hooks for comparing the current dataset version against a previous one.
 *
 * useComparisonFilteredRows — runs the filter chain on the comparison dataset ONCE.
 * useComparisonRanks        — computes rank deltas using pre-filtered comparison rows.
 */

import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { filterData } from '../data/filterChain';
import { computeRankDeltas, resolveComparisonVersion } from '../data/comparison';
import type { Row, RankDelta, FilterParams, ComparisonInfo } from '../data/types';

export interface ComparisonResult {
  /** Rank deltas keyed by compound reactant key (e.g. "PdCl2 / DMF") */
  rankMap: Map<string, RankDelta>;
  /** Per-axis rank deltas for heatmaps: one map per individual reactant type column */
  axisRankMaps: Map<string, RankDelta>[];
  /** Labels for the two datasets being compared */
  info: ComparisonInfo;
}

interface ComparisonFilterResult {
  rows: Row[];
  info: ComparisonInfo;
}

/** Format a version label from id + optional date. */
function formatVersionLabel(id: string, date?: string): string {
  return date ? `${id} (${date})` : id;
}

/**
 * Run the filter chain on the comparison dataset once.
 *
 * Returns the filtered comparison rows + version labels, or null when
 * comparison mode is off or comparison data is unavailable.
 */
export function useComparisonFilteredRows(): ComparisonFilterResult | null {
  const comparisonMode = useFilterStore((s) => s.comparisonMode);
  const comparisonVersion = useFilterStore((s) => s.comparisonVersion);
  const activeVersion = useFilterStore((s) => s.activeVersion);
  const availableVersions = useFilterStore((s) => s.availableVersions);
  const datasetCache = useFilterStore((s) => s.datasetCache);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);

  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const fgA = useFilterStore((s) => s.fgA);
  const fgB = useFilterStore((s) => s.fgB);
  const excludeCui = useFilterStore((s) => s.excludeCui);
  const excludeScaleup = useFilterStore((s) => s.excludeScaleup);
  const includeNullCategories = useFilterStore((s) => s.includeNullCategories);
  const minEln = useFilterStore((s) => s.minEln);
  const topnZscore = useFilterStore((s) => s.topnZscore);
  const maxComponents = useFilterStore((s) => s.maxComponents);

  const comparisonVersionId = useMemo(() => {
    if (!comparisonMode) return null;
    return resolveComparisonVersion(availableVersions, activeVersion, comparisonVersion, !!uploadedDataset);
  }, [comparisonMode, comparisonVersion, activeVersion, availableVersions, uploadedDataset]);

  const info = useMemo((): ComparisonInfo | null => {
    if (!comparisonVersionId) return null;
    const activeV = availableVersions.find((v) => v.id === activeVersion);
    const compV = availableVersions.find((v) => v.id === comparisonVersionId);
    return {
      currentLabel: activeV ? formatVersionLabel(activeV.label, activeV.date) : activeVersion,
      comparisonLabel: compV ? formatVersionLabel(compV.label, compV.date) : comparisonVersionId,
    };
  }, [comparisonVersionId, activeVersion, availableVersions]);

  const rawComparisonRows = useMemo(() => {
    if (!comparisonVersionId) return null;
    return datasetCache[comparisonVersionId]?.rows ?? null;
  }, [comparisonVersionId, datasetCache]);

  return useMemo(() => {
    if (!comparisonMode || !rawComparisonRows || !info) return null;

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

    return { rows: filterData(rawComparisonRows, params).rows, info };
  }, [
    comparisonMode,
    rawComparisonRows,
    info,
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
  ]);
}

/**
 * Compute rank deltas for a single panel using pre-filtered comparison rows.
 *
 * This is cheap (just grouping + median + sorting) — no filter chain involved.
 */
export function useComparisonRanks(
  currentRows: Row[],
  reactantTypes: string[],
  comparisonResult: ComparisonFilterResult | null,
): ComparisonResult | null {
  return useMemo(() => {
    if (!comparisonResult || reactantTypes.length === 0) return null;

    const { rows: comparisonFilteredRows, info } = comparisonResult;

    // Compound-key rank deltas (for boxplot/violin/stats)
    const rankMap = computeRankDeltas(currentRows, comparisonFilteredRows, reactantTypes);

    // Per-axis rank deltas (for heatmap axes)
    const axisRankMaps = reactantTypes.map((col) =>
      computeRankDeltas(currentRows, comparisonFilteredRows, [col]),
    );

    return { rankMap, axisRankMaps, info };
  }, [comparisonResult, currentRows, reactantTypes]);
}
