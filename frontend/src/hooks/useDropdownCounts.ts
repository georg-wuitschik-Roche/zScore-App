/**
 * Hook returning the per-option ELN counts for the four filter dropdowns.
 *
 * Each map applies every active filter except the one its own dropdown
 * controls, so a count says how many ELNs that option adds on top of the
 * current selection.
 */

import { useMemo } from 'react';
import {
  getReactionTypeElnCounts,
  getFgAElnCounts,
  getFgBElnCounts,
  getReactantElnCounts,
} from '../data/dropdownOptions';
import { useEffectiveDataset } from './useEffectiveDataset';
import { useFilterParams } from './useFilterParams';

export interface DropdownCounts {
  reactionTypes: Record<string, number>;
  fgA: Record<string, number>;
  fgB: Record<string, number>;
  reactantTypes: Record<string, number>;
}

export function useDropdownCounts(): DropdownCounts {
  const sourceData = useEffectiveDataset();
  const params = useFilterParams();

  return useMemo(
    () => ({
      reactionTypes: getReactionTypeElnCounts(sourceData, params),
      fgA: getFgAElnCounts(sourceData, params),
      fgB: getFgBElnCounts(sourceData, params),
      reactantTypes: getReactantElnCounts(sourceData, params),
    }),
    [sourceData, params],
  );
}
