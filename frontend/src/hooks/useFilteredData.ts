/**
 * Hook that returns filtered data derived from the current filter state.
 *
 * Uses useMemo so the filter chain only re-runs when filter params change.
 * This replaces the server-side filter_data() + LRU cache.
 */

import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { filterData } from '../data/filterChain';
import { useEffectiveDataset } from './useEffectiveDataset';
import type { FilterParams, FilterStats, Row } from '../data/types';

export interface FilteredResult {
  rows: Row[];
  stats: FilterStats;
}

export function useFilteredData(): FilteredResult {
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

  const sourceData = useEffectiveDataset();

  return useMemo(() => {
    if (sourceData.length === 0) {
      return { rows: [], stats: {} };
    }

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

    return filterData(sourceData, params);
  }, [
    sourceData,
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
