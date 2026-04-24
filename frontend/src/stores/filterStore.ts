/**
 * Zustand store for all filter state + dataset.
 *
 * This replaces Dash's dcc.Store components and callback state management.
 * All filter changes are synchronous and trigger React re-renders via useMemo.
 */

import { create } from 'zustand';
import type { Row, DropdownIndex, SplitSelector, TabId, VersionInfo, UploadMode, CatalystFilterMode } from '../data/types';
import { DEFAULTS, REQUIRED_COLUMNS, isTabId } from '../data/types';
import {
  fetchDropdownIndex,
  fetchParquetBuffer,
  parseDataset,
  parseCSVText,
  fetchVersionsManifest,
} from '../data/loader';
import { saveUpload, loadUpload, clearUpload as clearStoredUpload } from '../data/uploadStorage';


export interface FilterState {
  // Data
  dataset: Row[];
  uploadedDataset: Row[] | null;
  isFullDataLoaded: boolean;
  dropdownIndex: DropdownIndex | null;
  loadError: string | null;
  datasetCache: Record<string, { rows: Row[]; index: DropdownIndex }>;

  // Version management
  availableVersions: VersionInfo[];
  activeVersion: string;
  isLoadingVersion: boolean;

  // Upload mode
  uploadMode: UploadMode;

  // Filter controls
  reactionTypes: string[];
  reactantTypes: string[];
  fgA: string[];
  fgB: string[];
  copperFilter: CatalystFilterMode;
  precomplexedFilter: CatalystFilterMode;
  excludeScaleup: boolean;
  includeNullCategories: boolean;
  minEln: number;
  topnZscore: number;
  maxComponents: number;

  // Split mode
  splitSelector: SplitSelector | null;

  // Cross-filter (split-panel interactive filtering)
  crossFilterSelections: Record<string, string[]>;
  crossFilterOrder: string[];

  // Version comparison
  comparisonMode: boolean;
  comparisonVersion: string | null; // null = auto (previous version)

  // UI state
  activeTab: TabId;
  presentationMode: boolean;
  optionsPanelOpen: boolean;
  showElnLegend: boolean;
  theme: 'light' | 'dark';
  themePreference: 'light' | 'dark' | 'auto';
  uploadError: string | null;
  uploadFileName: string | null;

  // Actions
  setReactionTypes: (types: string[]) => void;
  setReactantTypes: (types: string[]) => void;
  setFgA: (fgs: string[]) => void;
  setFgB: (fgs: string[]) => void;
  setCopperFilter: (val: CatalystFilterMode) => void;
  setPrecomplexedFilter: (val: CatalystFilterMode) => void;
  setExcludeScaleup: (val: boolean) => void;
  setIncludeNullCategories: (val: boolean) => void;
  setMinEln: (val: number) => void;
  setTopnZscore: (val: number) => void;
  setMaxComponents: (val: number) => void;
  setSplitSelector: (selector: SplitSelector | null) => void;
  toggleCrossFilterValue: (panel: string, value: string, multi: boolean) => void;
  clearCrossFilters: () => void;
  setActiveTab: (tab: TabId) => void;
  togglePresentationMode: () => void;
  toggleOptionsPanel: () => void;
  toggleElnLegend: () => void;
  resetFilters: () => void;
  clearUploadError: () => void;
  loadDataset: () => Promise<void>;
  uploadCSV: (text: string, fileName?: string, mode?: UploadMode) => Promise<void>;
  switchVersion: (versionId: string) => Promise<void>;
  setUploadMode: (mode: UploadMode) => void;
  clearUploadData: () => void;
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  setComparisonMode: (on: boolean) => void;
  setComparisonVersion: (versionId: string | null) => void;
  resetOptions: () => void;

  // Bulk update (for URL state restoration)
  setFilters: (partial: Partial<FilterState>) => void;
}

/** Fetch, parse, and cache a single dataset version. Returns the cached entry. */
async function fetchAndCacheVersion(
  version: VersionInfo,
  set: (partial: Partial<FilterState> | ((s: FilterState) => Partial<FilterState>)) => void,
  get: () => FilterState,
): Promise<{ rows: Row[]; index: DropdownIndex }> {
  const existing = get().datasetCache[version.id];
  if (existing) return existing;

  const [index, buffer] = await Promise.all([
    fetchDropdownIndex(version.index),
    fetchParquetBuffer(version.parquet),
  ]);
  const rows = await parseDataset(buffer);
  const entry = { rows, index };
  set((s) => ({ datasetCache: { ...s.datasetCache, [version.id]: entry } }));
  return entry;
}

function getSystemTheme(): 'light' | 'dark' {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function resolveTheme(pref: 'light' | 'dark' | 'auto'): 'light' | 'dark' {
  return pref === 'auto' ? getSystemTheme() : pref;
}

const storedThemePref = (typeof localStorage !== 'undefined' && localStorage.getItem('zscore-theme') as 'light' | 'dark' | 'auto' | null) || 'auto';
const storedTab = (typeof localStorage !== 'undefined' && localStorage.getItem('zscore-tab')) || null;
const initialTab: TabId = storedTab && isTabId(storedTab) ? storedTab : 'violin';
const storedElnLegend = typeof localStorage !== 'undefined' && localStorage.getItem('zscore-eln-legend');
const initialElnLegend = storedElnLegend !== null ? storedElnLegend !== '0' : true;

export const useFilterStore = create<FilterState>((set, get) => ({
  // Data
  dataset: [],
  uploadedDataset: null,
  isFullDataLoaded: false,
  dropdownIndex: null,
  loadError: null,
  datasetCache: {},

  // Version management
  availableVersions: [],
  activeVersion: '',
  isLoadingVersion: false,

  // Upload mode & persistence
  uploadMode: 'replace',

  // Default filter values
  reactionTypes: DEFAULTS.reactionTypes,
  reactantTypes: DEFAULTS.reactantTypes,
  fgA: DEFAULTS.fgA,
  fgB: DEFAULTS.fgB,
  copperFilter: DEFAULTS.copperFilter,
  precomplexedFilter: DEFAULTS.precomplexedFilter,
  excludeScaleup: DEFAULTS.excludeScaleup,
  includeNullCategories: DEFAULTS.includeNullCategories,
  minEln: DEFAULTS.minEln,
  topnZscore: DEFAULTS.topnZscore,
  maxComponents: DEFAULTS.maxComponents,

  // Version comparison
  comparisonMode: false,
  comparisonVersion: null,

  // Split mode
  splitSelector: null,
  crossFilterSelections: {},
  crossFilterOrder: [],

  // UI state
  activeTab: initialTab,
  presentationMode: false,
  optionsPanelOpen: false,
  showElnLegend: initialElnLegend,
  themePreference: storedThemePref,
  theme: resolveTheme(storedThemePref),
  uploadError: null,
  uploadFileName: null,

  // Actions
  setReactionTypes: (types) =>
    set((s) => ({
      reactionTypes: types,
      fgA: [],
      fgB: [],
      crossFilterSelections: {}, crossFilterOrder: [],
      // Auto-clear split if it was on reactionTypes (now <2) or fgA/fgB (just emptied)
      splitSelector:
        (s.splitSelector === 'reactionTypes' && types.length < 2) ||
        s.splitSelector === 'fgA' ||
        s.splitSelector === 'fgB'
          ? null
          : s.splitSelector,
    })),
  setReactantTypes: (types) =>
    set((s) => ({
      reactantTypes: types,
      crossFilterSelections: {}, crossFilterOrder: [],
      splitSelector:
        s.splitSelector === 'reactantTypes' && types.length < 2
          ? null
          : s.splitSelector,
    })),
  setFgA: (fgs) =>
    set((s) => ({
      fgA: fgs,
      splitSelector:
        s.splitSelector === 'fgA' && fgs.length < 2 ? null : s.splitSelector,
    })),
  setFgB: (fgs) =>
    set((s) => ({
      fgB: fgs,
      splitSelector:
        s.splitSelector === 'fgB' && fgs.length < 2 ? null : s.splitSelector,
    })),
  setCopperFilter: (val) => set({ copperFilter: val }),
  setPrecomplexedFilter: (val) => set({ precomplexedFilter: val }),
  setExcludeScaleup: (val) => set({ excludeScaleup: val }),
  setIncludeNullCategories: (val) => set({ includeNullCategories: val }),
  setMinEln: (val) => set({ minEln: val }),
  setTopnZscore: (val) => set({ topnZscore: val }),
  setMaxComponents: (val) => set({ maxComponents: val }),
  setSplitSelector: (selector) => set({ splitSelector: selector, crossFilterSelections: {}, crossFilterOrder: [] }),
  toggleCrossFilterValue: (panel, value, multi) =>
    set((s) => {
      const prev = s.crossFilterSelections[panel] ?? [];
      let next: string[];
      if (multi) {
        next = prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value];
      } else {
        next = prev.length === 1 && prev[0] === value ? [] : [value];
      }
      const selections = { ...s.crossFilterSelections };
      if (next.length === 0) {
        delete selections[panel];
      } else {
        selections[panel] = next;
      }
      const order = next.length === 0
        ? s.crossFilterOrder.filter((p) => p !== panel)
        : s.crossFilterOrder.includes(panel) ? s.crossFilterOrder : [...s.crossFilterOrder, panel];
      return { crossFilterSelections: selections, crossFilterOrder: order };
    }),
  clearCrossFilters: () => set({ crossFilterSelections: {}, crossFilterOrder: [] }),
  setActiveTab: (tab) => {
    if (tab === get().activeTab) return;
    try { localStorage.setItem('zscore-tab', tab); } catch { /* ignore */ }
    set({ activeTab: tab });
  },
  togglePresentationMode: () =>
    set((s) => ({ presentationMode: !s.presentationMode })),
  toggleOptionsPanel: () =>
    set((s) => ({ optionsPanelOpen: !s.optionsPanelOpen })),
  toggleElnLegend: () => {
    const next = !get().showElnLegend;
    try { localStorage.setItem('zscore-eln-legend', next ? '1' : '0'); } catch { /* ignore */ }
    set({ showElnLegend: next });
  },
  setTheme: (pref) => {
    const resolved = resolveTheme(pref);
    document.documentElement.setAttribute('data-theme', resolved);
    try { localStorage.setItem('zscore-theme', pref); } catch { /* ignore */ }
    set({ themePreference: pref, theme: resolved });
  },

  resetFilters: () => {
    clearStoredUpload();
    set({
      ...DEFAULTS,
      splitSelector: null,
      crossFilterSelections: {}, crossFilterOrder: [],
      activeTab: initialTab,
      uploadedDataset: null,
      uploadFileName: null,
      uploadMode: 'replace',
      comparisonMode: false,
      comparisonVersion: null,
        });
  },

  loadDataset: async () => {
    set({ loadError: null });

    // Discover available dataset versions
    const manifest = await fetchVersionsManifest();
    const latest = manifest.versions.find((v) => v.id === manifest.latest) ?? manifest.versions[0];
    set({ availableVersions: manifest.versions, activeVersion: latest.id });

    // Phase 1+2: Fetch dropdown index + parquet, cache, and activate
    try {
      const entry = await fetchAndCacheVersion(latest, set, get);
      set({ dataset: entry.rows, dropdownIndex: entry.index, isFullDataLoaded: true });
    } catch (e) {
      set({
        loadError: e instanceof Error ? e.message : 'Failed to load dataset',
      });
    }

    // Phase 3: Preload remaining dataset versions in the background
    for (const version of manifest.versions) {
      if (version.id !== latest.id) {
        fetchAndCacheVersion(version, set, get).catch(() => {/* preload failure is non-critical */});
      }
    }

    // Phase 4: Restore persisted upload from localStorage
    const stored = loadUpload();
    if (stored) {
      set({
        uploadedDataset: stored.rows,
        uploadFileName: stored.fileName,
        uploadMode: stored.mode,
      });
    }
  },

  uploadCSV: async (text, fileName, mode) => {
    const uploadMode = mode ?? get().uploadMode;
    try {
      const rows = await parseCSVText(text);
      if (rows.length === 0) {
        set({ uploadError: 'The uploaded file contains no data rows.' });
        return;
      }

      const columns = Object.keys(rows[0]);
      const missing = REQUIRED_COLUMNS.filter((c) => !columns.includes(c));
      if (missing.length > 0) {
        set({ uploadError: `Missing required columns: ${missing.join(', ')}` });
        return;
      }

      // Check z-Score has numeric values
      const hasNumericZScore = rows.some((r) => {
        const z = r['z-Score'];
        return z !== null && z !== undefined && !isNaN(Number(z));
      });
      if (!hasNumericZScore) {
        set({ uploadError: 'Column "z-Score" contains no valid numeric values.' });
        return;
      }

      // Prefix ELN IDs with upload_ to avoid collisions when combining
      if (uploadMode === 'combine') {
        for (const row of rows) {
          if (row.ELN_ID && !row.ELN_ID.startsWith('upload_')) {
            row.ELN_ID = `upload_${row.ELN_ID}`;
          }
        }
      }

      const name = fileName ?? null;
      saveUpload(rows, name ?? 'upload.csv', uploadMode);
      set({
        uploadedDataset: rows,
        uploadError: null,
        uploadFileName: name,
      });
    } catch {
      set({ uploadError: 'Failed to parse CSV file. Check the format and encoding.' });
    }
  },

  switchVersion: async (versionId) => {
    const { availableVersions } = get();
    const version = availableVersions.find((v) => v.id === versionId);
    if (!version) return;

    const isCached = !!get().datasetCache[versionId];
    if (!isCached) set({ isLoadingVersion: true, loadError: null });

    try {
      const entry = await fetchAndCacheVersion(version, set, get);
      set({
        dataset: entry.rows,
        dropdownIndex: entry.index,
        isFullDataLoaded: true,
        activeVersion: versionId,
        isLoadingVersion: false,
        comparisonVersion: null,
      });
    } catch (e) {
      set({
        loadError: e instanceof Error ? e.message : 'Failed to load dataset version',
        isLoadingVersion: false,
      });
    }
  },

  setUploadMode: (mode) => {
    const { uploadMode: prev, uploadedDataset } = get();
    if (mode === prev) return;
    // Re-prefix ELN IDs when switching modes on existing upload
    if (uploadedDataset) {
      const rows = uploadedDataset.map((row) => {
        if (mode === 'combine' && row.ELN_ID && !row.ELN_ID.startsWith('upload_')) {
          return { ...row, ELN_ID: `upload_${row.ELN_ID}` };
        }
        if (mode === 'replace' && row.ELN_ID?.startsWith('upload_')) {
          return { ...row, ELN_ID: row.ELN_ID.slice(7) };
        }
        return row;
      });
      saveUpload(rows, get().uploadFileName ?? 'upload.csv', mode);
      set({ uploadMode: mode, uploadedDataset: rows });
    } else {
      set({ uploadMode: mode });
    }
  },

  clearUploadData: () => {
    clearStoredUpload();
    set({
      uploadedDataset: null,
      uploadFileName: null,
        });
  },

  resetOptions: () =>
    set({
      minEln: DEFAULTS.minEln,
      topnZscore: DEFAULTS.topnZscore,
      maxComponents: DEFAULTS.maxComponents,
      copperFilter: DEFAULTS.copperFilter,
      precomplexedFilter: DEFAULTS.precomplexedFilter,
      excludeScaleup: DEFAULTS.excludeScaleup,
      includeNullCategories: DEFAULTS.includeNullCategories,
    }),

  setComparisonMode: (on) => set({ comparisonMode: on }),
  setComparisonVersion: (versionId) => set({ comparisonVersion: versionId }),

  clearUploadError: () => set({ uploadError: null }),

  setFilters: (partial) => set(partial),
}));
