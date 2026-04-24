/**
 * Hook that returns filtered data split into panels when split mode is active.
 *
 * When no split is active, returns a single-element array identical to useFilteredData().
 * When split, runs the filter chain once per value in the split dimension.
 */

import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { filterData } from '../data/filterChain';
import { useEffectiveDataset } from './useEffectiveDataset';
import type { FilterParams, SplitPanel } from '../data/types';

export function useSplitFilteredData(): SplitPanel[] {
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
  const splitSelector = useFilterStore((s) => s.splitSelector);
  const crossFilterSelections = useFilterStore((s) => s.crossFilterSelections);
  const crossFilterOrder = useFilterStore((s) => s.crossFilterOrder);

  const sourceData = useEffectiveDataset();

  return useMemo(() => {
    if (sourceData.length === 0) {
      return [{ label: 'Combined', rows: [], stats: {}, reactantTypes }];
    }

    const baseParams: FilterParams = {
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
    const hasCrossFilter =
      splitSelector === 'reactantTypes' &&
      Object.keys(crossFilterSelections).length > 0;

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
      let { rows } = result;

      // Cross-filter: a panel receives filters from panels selected BEFORE it
      // (earlier in crossFilterOrder). The first-clicked panel is never filtered.
      // Panels without a selection receive from ALL selected panels.
      if (hasCrossFilter) {
        const myOrderIdx = crossFilterOrder.indexOf(value);
        const upstream = myOrderIdx < 0
          ? crossFilterOrder                    // no selection → receive from all
          : crossFilterOrder.slice(0, myOrderIdx); // receive only from earlier
        if (upstream.length > 0) {
          rows = rows.filter((row) => {
            for (const panel of upstream) {
              const sel = crossFilterSelections[panel];
              if (!sel || sel.length === 0) continue;
              if (!sel.includes(String(row[panel] ?? ''))) return false;
            }
            return true;
          });
        }
      }

      return {
        label: value,
        rows,
        stats: result.stats,
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
    copperFilter,
    precomplexedFilter,
    excludeScaleup,
    includeNullCategories,
    minEln,
    topnZscore,
    maxComponents,
    splitSelector,
    crossFilterSelections,
    crossFilterOrder,
  ]);
}
