/**
 * Hook that returns filtered data derived from the current filter state.
 *
 * Uses useMemo so the filter chain only re-runs when filter params change.
 * This replaces the server-side filter_data() + LRU cache.
 */

import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { filterData } from '../data/filterChain';
import type { FilterParams, FilterStats, Row } from '../data/types';

export interface FilteredResult {
  rows: Row[];
  stats: FilterStats;
}

export function useFilteredData(): FilteredResult {
  const {
    dataset,
    uploadedDataset,
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
  } = useFilterStore();

  const sourceData = uploadedDataset ?? dataset;

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
