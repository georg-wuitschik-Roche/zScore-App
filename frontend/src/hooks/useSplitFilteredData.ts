/**
 * Hook that returns filtered data split into panels when split mode is active.
 *
 * When no split is active, returns a single-element array identical to useFilteredData().
 * When split, runs the filter chain once per value in the split dimension.
 */

import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { filterData } from '../data/filterChain';
import type { FilterParams } from '../data/filterChain';
import type { Row, FilterStats } from '../data/types';

export interface SplitPanel {
  label: string;
  rows: Row[];
  stats: FilterStats;
  reactantTypes: string[];
}

export function useSplitFilteredData(): SplitPanel[] {
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
    splitSelector,
  } = useFilterStore();

  const sourceData = uploadedDataset ?? dataset;

  return useMemo(() => {
    if (sourceData.length === 0) {
      return [{ label: 'Combined', rows: [], stats: {}, reactantTypes }];
    }

    const baseParams: FilterParams = {
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

    // Determine which values to split on
    const splitValues = splitSelector
      ? {
          reactionTypes,
          fgA,
          fgB,
          reactantTypes,
        }[splitSelector]
      : null;

    // Fall back to combined if no split or <2 values
    if (!splitSelector || !splitValues || splitValues.length < 2) {
      const result = filterData(sourceData, baseParams);
      return [{ label: 'Combined', ...result, reactantTypes }];
    }

    // Run filter chain once per split value
    return splitValues.map((value) => {
      const params: FilterParams = { ...baseParams };

      if (splitSelector === 'reactantTypes') {
        params.reactantTypes = [value];
      } else if (splitSelector === 'reactionTypes') {
        params.reactionTypes = [value];
      } else if (splitSelector === 'fgA') {
        params.fgA = [value];
      } else {
        params.fgB = [value];
      }

      const result = filterData(sourceData, params);
      return {
        label: value,
        ...result,
        reactantTypes:
          splitSelector === 'reactantTypes' ? [value] : reactantTypes,
      };
    });
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
    splitSelector,
  ]);
}
