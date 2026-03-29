/**
 * Zustand store for all filter state + dataset.
 *
 * This replaces Dash's dcc.Store components and callback state management.
 * All filter changes are synchronous and trigger React re-renders via useMemo.
 */

import { create } from 'zustand';
import type { Row, DropdownIndex, SplitSelector, TabId, VersionInfo, UploadMode } from '../data/types';
import { REQUIRED_COLUMNS } from '../data/types';
import {
  fetchDropdownIndex,
  fetchParquetBuffer,
  parseDataset,
  parseCSVText,
  fetchVersionsManifest,
} from '../data/loader';
import { saveUpload, loadUpload, clearUpload as clearStoredUpload } from '../data/uploadStorage';

// Default filter values — empty until user selects on landing page
const DEFAULT_REACTION_TYPES: string[] = [];
const DEFAULT_FG_A: string[] = [];
const DEFAULT_FG_B: string[] = [];
const DEFAULT_REACTANT_TYPES: string[] = [];
const DEFAULT_MIN_ELN = 5;
const DEFAULT_TOPN_ZSCORE = 5;
const DEFAULT_MAX_COMPONENTS = 10;

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
  excludeCui: boolean;
  excludeScaleup: boolean;
  includeNullCategories: boolean;
  minEln: number;
  topnZscore: number;
  maxComponents: number;

  // Split mode
  splitSelector: SplitSelector | null;

  // Version comparison
  comparisonMode: boolean;
  comparisonVersion: string | null; // null = auto (previous version)

  // UI state
  activeTab: TabId;
  presentationMode: boolean;
  optionsPanelOpen: boolean;
  theme: 'light' | 'dark';
  themePreference: 'light' | 'dark' | 'auto';
  uploadError: string | null;
  uploadFileName: string | null;

  // Actions
  setReactionTypes: (types: string[]) => void;
  setReactantTypes: (types: string[]) => void;
  setFgA: (fgs: string[]) => void;
  setFgB: (fgs: string[]) => void;
  setExcludeCui: (val: boolean) => void;
  setExcludeScaleup: (val: boolean) => void;
  setIncludeNullCategories: (val: boolean) => void;
  setMinEln: (val: number) => void;
  setTopnZscore: (val: number) => void;
  setMaxComponents: (val: number) => void;
  setSplitSelector: (selector: SplitSelector | null) => void;
  setActiveTab: (tab: TabId) => void;
  togglePresentationMode: () => void;
  toggleOptionsPanel: () => void;
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
  reactionTypes: DEFAULT_REACTION_TYPES,
  reactantTypes: DEFAULT_REACTANT_TYPES,
  fgA: DEFAULT_FG_A,
  fgB: DEFAULT_FG_B,
  excludeCui: true,
  excludeScaleup: true,
  includeNullCategories: true,
  minEln: DEFAULT_MIN_ELN,
  topnZscore: DEFAULT_TOPN_ZSCORE,
  maxComponents: DEFAULT_MAX_COMPONENTS,

  // Version comparison
  comparisonMode: false,
  comparisonVersion: null,

  // Split mode
  splitSelector: null,

  // UI state
  activeTab: 'boxplot',
  presentationMode: false,
  optionsPanelOpen: false,
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
  setExcludeCui: (val) => set({ excludeCui: val }),
  setExcludeScaleup: (val) => set({ excludeScaleup: val }),
  setIncludeNullCategories: (val) => set({ includeNullCategories: val }),
  setMinEln: (val) => set({ minEln: val }),
  setTopnZscore: (val) => set({ topnZscore: val }),
  setMaxComponents: (val) => set({ maxComponents: val }),
  setSplitSelector: (selector) => set({ splitSelector: selector }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  togglePresentationMode: () =>
    set((s) => ({ presentationMode: !s.presentationMode })),
  toggleOptionsPanel: () =>
    set((s) => ({ optionsPanelOpen: !s.optionsPanelOpen })),
  setTheme: (pref) => {
    const resolved = resolveTheme(pref);
    document.documentElement.setAttribute('data-theme', resolved);
    try { localStorage.setItem('zscore-theme', pref); } catch { /* ignore */ }
    set({ themePreference: pref, theme: resolved });
  },

  resetFilters: () => {
    clearStoredUpload();
    set({
      reactionTypes: [],
      reactantTypes: [],
      fgA: [],
      fgB: [],
      excludeCui: true,
      excludeScaleup: true,
      includeNullCategories: true,
      minEln: DEFAULT_MIN_ELN,
      topnZscore: DEFAULT_TOPN_ZSCORE,
      maxComponents: DEFAULT_MAX_COMPONENTS,
      splitSelector: null,
      activeTab: 'boxplot',
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
      minEln: DEFAULT_MIN_ELN,
      topnZscore: DEFAULT_TOPN_ZSCORE,
      maxComponents: DEFAULT_MAX_COMPONENTS,
      excludeCui: true,
      excludeScaleup: true,
      includeNullCategories: true,
    }),

  setComparisonMode: (on) => set({ comparisonMode: on }),
  setComparisonVersion: (versionId) => set({ comparisonVersion: versionId }),

  clearUploadError: () => set({ uploadError: null }),

  setFilters: (partial) => set(partial),
}));

export { DEFAULT_REACTION_TYPES, DEFAULT_FG_A, DEFAULT_FG_B, DEFAULT_REACTANT_TYPES };
export { DEFAULT_MIN_ELN, DEFAULT_TOPN_ZSCORE, DEFAULT_MAX_COMPONENTS };
