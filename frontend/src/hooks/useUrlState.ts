/**
 * Bidirectional sync between Zustand filter store and URL search params.
 *
 * - Filter changes → URL updates (debounced 250ms)
 * - URL changes (back/forward) → filter state restoration
 */

import { useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import type { FilterState } from '../stores/filterStore'; // used for Partial<FilterState>
import { SPLIT_URL_KEYS, COPPER_FILTER_OPTIONS } from '../data/types';
import type { SplitSelector, CopperFilter } from '../data/types';

const URL_TO_SPLIT = Object.fromEntries(
  Object.entries(SPLIT_URL_KEYS).map(([k, v]) => [v, k]),
) as Record<string, SplitSelector>;

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

  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const fgA = useFilterStore((s) => s.fgA);
  const fgB = useFilterStore((s) => s.fgB);
  const copperFilter = useFilterStore((s) => s.copperFilter);
  const excludeScaleup = useFilterStore((s) => s.excludeScaleup);
  const includeNullCategories = useFilterStore((s) => s.includeNullCategories);
  const minEln = useFilterStore((s) => s.minEln);
  const topnZscore = useFilterStore((s) => s.topnZscore);
  const maxComponents = useFilterStore((s) => s.maxComponents);
  const activeTab = useFilterStore((s) => s.activeTab);
  const splitSelector = useFilterStore((s) => s.splitSelector);
  const crossFilterSelections = useFilterStore((s) => s.crossFilterSelections);
  const crossFilterOrder = useFilterStore((s) => s.crossFilterOrder);
  const activeVersion = useFilterStore((s) => s.activeVersion);
  const comparisonMode = useFilterStore((s) => s.comparisonMode);
  const comparisonVersion = useFilterStore((s) => s.comparisonVersion);
  const switchVersion = useFilterStore((s) => s.switchVersion);
  const setFilters = useFilterStore((s) => s.setFilters);

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
    const cu = searchParams.get('cu');
    const su = searchParams.get('su');
    const nc = searchParams.get('nc');
    const split = searchParams.get('split');
    const ver = searchParams.get('ver');
    const cmp = searchParams.get('cmp');
    const cmpv = searchParams.get('cmpv');
    const xf = searchParams.get('xf');

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
    if (cu !== null) {
      partial.copperFilter = COPPER_FILTER_OPTIONS.includes(cu as CopperFilter) ? (cu as CopperFilter) : 'exclude';
    }
    if (su !== null) partial.excludeScaleup = su === '1';
    if (nc !== null) partial.includeNullCategories = nc === '1';
    partial.splitSelector = split ? (URL_TO_SPLIT[split] ?? null) : null;
    if (cmp !== null) partial.comparisonMode = cmp === '1';
    if (cmpv) partial.comparisonVersion = cmpv;
    if (xf) {
      const selections: Record<string, string[]> = {};
      for (const part of xf.split(';')) {
        const colonIdx = part.indexOf(':');
        if (colonIdx < 1) continue;
        const panel = part.slice(0, colonIdx);
        const vals = part.slice(colonIdx + 1).split('|').filter(Boolean);
        if (vals.length > 0) selections[panel] = vals;
      }
      partial.crossFilterSelections = selections;
      partial.crossFilterOrder = Object.keys(selections);
    }

    setFilters(partial);

    // Switch version if URL specifies one different from current
    if (ver && ver !== activeVersion) {
      switchVersion(ver);
    }

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
      params.set('cu', copperFilter);
      params.set('su', excludeScaleup ? '1' : '0');
      params.set('nc', includeNullCategories ? '1' : '0');
      if (activeTab !== 'violin') params.set('tab', activeTab);
      if (splitSelector) params.set('split', SPLIT_URL_KEYS[splitSelector]);
      if (activeVersion && activeVersion !== 'default') params.set('ver', activeVersion);
      if (comparisonMode) {
        params.set('cmp', '1');
        if (comparisonVersion) params.set('cmpv', comparisonVersion);
      }
      if (crossFilterOrder.length > 0) {
        const xfStr = crossFilterOrder
          .filter((p) => crossFilterSelections[p]?.length > 0)
          .map((p) => `${p}:${crossFilterSelections[p].join('|')}`)
          .join(';');
        if (xfStr) params.set('xf', xfStr);
      }

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
    copperFilter,
    excludeScaleup,
    includeNullCategories,
    minEln,
    topnZscore,
    maxComponents,
    activeTab,
    splitSelector,
    crossFilterSelections,
    crossFilterOrder,
    activeVersion,
    comparisonMode,
    comparisonVersion,
  ]);
}
