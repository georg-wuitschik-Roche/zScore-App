/**
 * Bidirectional sync between Zustand filter store and URL search params.
 *
 * - Filter changes → URL updates (debounced 250ms)
 * - URL changes (back/forward) → filter state restoration
 */

import { useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import type { FilterState } from '../stores/filterStore';

/** Serialize a string array to a URL param (pipe-separated to avoid comma conflicts). */
function encodeArray(arr: string[]): string {
  return arr.join('|');
}

/** Deserialize a URL param to a string array. */
function decodeArray(val: string | null): string[] | null {
  if (!val) return null;
  // Always use pipe separator — commas appear in reaction type names
  // (e.g., "Borylation, Miyaura", "Negishi, in-situ")
  return val.split('|').filter(Boolean);
}

export function useUrlState(): void {
  const [searchParams, setSearchParams] = useSearchParams();
  const isRestoringRef = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const {
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
    activeTab,
    setFilters,
  } = useFilterStore((s) => s);

  // Restore from URL on mount (or browser back/forward)
  useEffect(() => {
    const rt = decodeArray(searchParams.get('rt'));
    const cat = decodeArray(searchParams.get('cat'));
    const fga = decodeArray(searchParams.get('fga'));
    const fgb = decodeArray(searchParams.get('fgb'));
    const me = searchParams.get('me');
    const tn = searchParams.get('tn');
    const mc = searchParams.get('mc');
    const tab = searchParams.get('tab');
    const cui = searchParams.get('cui');
    const su = searchParams.get('su');
    const nc = searchParams.get('nc');

    // Only restore if URL has params
    if (!rt && !cat && !fga && !fgb && !me) return;

    isRestoringRef.current = true;

    const partial: Partial<FilterState> = {};
    if (rt) partial.reactionTypes = rt;
    if (cat) partial.reactantTypes = cat;
    if (fga) partial.fgA = fga;
    if (fgb) partial.fgB = fgb;
    if (me) partial.minEln = Number(me);
    if (tn) partial.topnZscore = Number(tn);
    if (mc) partial.maxComponents = Number(mc);
    if (tab) partial.activeTab = tab as 'boxplot' | 'violin' | 'heatmap' | 'stats';
    if (cui !== null) partial.excludeCui = cui === '1';
    if (su !== null) partial.excludeScaleup = su === '1';
    if (nc !== null) partial.includeNullCategories = nc === '1';

    setFilters(partial);

    // Clear restore flag after a tick
    setTimeout(() => {
      isRestoringRef.current = false;
    }, 100);
    // Only run on mount and when URL changes via browser navigation
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  // Push filter state to URL (debounced)
  useEffect(() => {
    if (isRestoringRef.current) return;

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const params = new URLSearchParams();
      if (reactionTypes.length > 0) params.set('rt', encodeArray(reactionTypes));
      if (reactantTypes.length > 0) params.set('cat', encodeArray(reactantTypes));
      if (fgA.length > 0) params.set('fga', encodeArray(fgA));
      if (fgB.length > 0) params.set('fgb', encodeArray(fgB));
      params.set('me', String(minEln));
      params.set('tn', String(topnZscore));
      params.set('mc', String(maxComponents));
      params.set('cui', excludeCui ? '1' : '0');
      params.set('su', excludeScaleup ? '1' : '0');
      params.set('nc', includeNullCategories ? '1' : '0');
      if (activeTab !== 'boxplot') params.set('tab', activeTab);

      setSearchParams(params, { replace: true });
    }, 250);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
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
    activeTab,
  ]);
}
