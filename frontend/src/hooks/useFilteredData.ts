/**
 * Hook that returns filtered data derived from the current filter state.
 *
 * Uses useMemo so the filter chain only re-runs when filter params change.
 * This replaces the server-side filter_data() + LRU cache.
 */

import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { filterData } from '../data/filterChain';
import type { FilterParams } from '../data/filterChain';
import type { Row } from '../data/types';

export interface FilteredResult {
  rows: Row[];
  stats: {
    wholeDataset?: { elns: number };
    afterReactantFilters?: { elns: number };
    afterFgA?: { elns: number };
    afterFgB?: { elns: number };
    maxComponentsCap?: number;
  };
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
