/**
 * Hook that returns filtered data derived from the current filter state.
 *
 * Uses useMemo so the filter chain only re-runs when filter params change.
 * This replaces the server-side filter_data() + LRU cache.
 */

import { useMemo } from 'react';
import { filterData } from '../data/filterChain';
import { useEffectiveDataset } from './useEffectiveDataset';
import { useFilterParams } from './useFilterParams';
import type { FilterStats, Row } from '../data/types';

export interface FilteredResult {
  rows: Row[];
  stats: FilterStats;
}

export function useFilteredData(): FilteredResult {
  const params = useFilterParams();
  const sourceData = useEffectiveDataset();

  return useMemo(() => {
    if (sourceData.length === 0) {
      return { rows: [], stats: {} };
    }
    return filterData(sourceData, params);
  }, [sourceData, params]);
}
