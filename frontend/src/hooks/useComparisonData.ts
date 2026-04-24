/**
 * Hooks for comparing the current dataset version against a previous one.
 *
 * useComparisonRawData — resolves comparison version, returns raw rows + filter params.
 * useComparisonRanks   — filters comparison rows per-panel and computes rank deltas.
 *
 * Filtering happens per-panel (not globally) so that split panels get comparison
 * data filtered with the panel's specific reactantTypes — otherwise maxComponents
 * and minEln grouping would use compound keys that don't match the panel's grouping.
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

export interface ComparisonRawData {
  /** Unfiltered rows from the comparison version */
  rawRows: Row[];
  /** Base filter params (reactantTypes will be overridden per-panel) */
  baseParams: FilterParams;
  /** Labels for the two datasets being compared */
  info: ComparisonInfo;
}

/** Format a version label from id + optional date. */
function formatVersionLabel(id: string, date?: string): string {
  return date ? `${id} (${date})` : id;
}

/**
 * Resolve the comparison version and return the raw (unfiltered) comparison rows
 * along with the base filter params and version labels.
 *
 * Filtering is deferred to useComparisonRanks so each split panel can apply
 * its own reactantTypes to the filter chain.
 */
export function useComparisonRawData(): ComparisonRawData | null {
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
  const copperFilter = useFilterStore((s) => s.copperFilter);
  const precomplexedFilter = useFilterStore((s) => s.precomplexedFilter);
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

  const baseParams = useMemo((): FilterParams => ({
    reactionTypes,
    reactantTypes,
    fgA,
    fgB,
    copperFilter,
    precomplexedFilter,
    excludeScaleup,
    includeNullCategories,
    minEln,
    topnZscore,
    maxComponents,
  }), [
    reactionTypes, reactantTypes, fgA, fgB,
    copperFilter, precomplexedFilter, excludeScaleup, includeNullCategories,
    minEln, topnZscore, maxComponents,
  ]);

  return useMemo(() => {
    if (!comparisonMode || !rawComparisonRows || !info) return null;
    return { rawRows: rawComparisonRows, baseParams, info };
  }, [comparisonMode, rawComparisonRows, baseParams, info]);
}

/**
 * Filter comparison rows for a specific panel and compute rank deltas.
 *
 * Runs the filter chain with the panel's reactantTypes (which may differ from
 * the store's reactantTypes when split mode is active). This ensures that
 * maxComponents, minEln, and topN grouping match the panel's grouping.
 */
export function useComparisonRanks(
  currentRows: Row[],
  reactantTypes: string[],
  comparisonRawData: ComparisonRawData | null,
): ComparisonResult | null {
  return useMemo(() => {
    if (!comparisonRawData || reactantTypes.length === 0) return null;

    const { rawRows, baseParams, info } = comparisonRawData;

    // Filter comparison rows with this panel's reactantTypes
    const panelParams: FilterParams = { ...baseParams, reactantTypes };
    const { rows: comparisonFilteredRows } = filterData(rawRows, panelParams);

    // Compound-key rank deltas (for boxplot/violin/stats)
    const rankMap = computeRankDeltas(currentRows, comparisonFilteredRows, reactantTypes);

    // Per-axis rank deltas (for heatmap axes)
    const axisRankMaps = reactantTypes.map((col) =>
      computeRankDeltas(currentRows, comparisonFilteredRows, [col]),
    );

    return { rankMap, axisRankMaps, info };
  }, [comparisonRawData, currentRows, reactantTypes]);
}
