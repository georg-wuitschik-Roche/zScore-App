/**
 * Hook that assembles the current filter state into a FilterParams object.
 *
 * Shared by the filter chain and the dropdown ELN counts so both read the
 * same state through one definition.
 */

import { useShallow } from 'zustand/react/shallow';
import { useFilterStore } from '../stores/filterStore';
import type { FilterParams } from '../data/types';

export function useFilterParams(): FilterParams {
  return useFilterStore(
    useShallow((s) => ({
      reactionTypes: s.reactionTypes,
      reactantTypes: s.reactantTypes,
      fgA: s.fgA,
      fgB: s.fgB,
      copperFilter: s.copperFilter,
      precomplexedFilter: s.precomplexedFilter,
      excludeScaleup: s.excludeScaleup,
      includeNullCategories: s.includeNullCategories,
      minEln: s.minEln,
      topnZscore: s.topnZscore,
      maxComponents: s.maxComponents,
    })),
  );
}
