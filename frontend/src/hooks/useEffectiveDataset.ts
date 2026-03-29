/**
 * Centralized hook for the effective dataset (built-in, uploaded, or combined).
 *
 * Replaces the scattered `uploadedDataset ?? dataset` pattern.
 */

import { useMemo } from 'react';
import type { Row } from '../data/types';
import { useFilterStore } from '../stores/filterStore';

export function useEffectiveDataset(): Row[] {
  const dataset = useFilterStore((s) => s.dataset);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const uploadMode = useFilterStore((s) => s.uploadMode);

  return useMemo(() => {
    if (!uploadedDataset) return dataset;
    if (uploadMode === 'replace') return uploadedDataset;
    // combine mode: concatenate built-in + uploaded
    return [...dataset, ...uploadedDataset];
  }, [dataset, uploadedDataset, uploadMode]);
}
