/**
 * Zustand store for all filter state + dataset.
 *
 * This replaces Dash's dcc.Store components and callback state management.
 * All filter changes are synchronous and trigger React re-renders via useMemo.
 */

import { create } from 'zustand';
import type { Row } from '../data/types';
import { loadDataset, parseCSVText } from '../data/loader';

// Default filter values — matches callbacks.py defaults
const DEFAULT_REACTION_TYPES = ['Buchwald-Hartwig'];
const DEFAULT_FG_A = ['RNH2 a-branch', 'RNH2'];
const DEFAULT_FG_B = ['ArBr', 'ArCl'];
const DEFAULT_REACTANT_TYPES = ['Catalyst'];
const DEFAULT_MIN_ELN = 5;
const DEFAULT_TOPN_ZSCORE = 5;
const DEFAULT_MAX_COMPONENTS = 10;

export interface FilterState {
  // Data
  dataset: Row[];
  uploadedDataset: Row[] | null;
  isLoading: boolean;
  loadError: string | null;

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

  // UI state
  activeTab: 'boxplot' | 'heatmap' | 'stats';
  presentationMode: boolean;
  optionsPanelOpen: boolean;
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
  setActiveTab: (tab: 'boxplot' | 'heatmap' | 'stats') => void;
  togglePresentationMode: () => void;
  toggleOptionsPanel: () => void;
  resetFilters: () => void;
  clearUploadError: () => void;
  loadDataset: () => Promise<void>;
  setUploadedDataset: (rows: Row[] | null) => void;
  uploadCSV: (text: string, fileName?: string) => void;

  // Bulk update (for URL state restoration)
  setFilters: (partial: Partial<FilterState>) => void;
}

export const useFilterStore = create<FilterState>((set) => ({
  // Data
  dataset: [],
  uploadedDataset: null,
  isLoading: true,
  loadError: null,

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

  // UI state
  activeTab: 'boxplot',
  presentationMode: false,
  optionsPanelOpen: false,
  uploadError: null,
  uploadFileName: null,

  // Actions
  setReactionTypes: (types) => set({ reactionTypes: types, fgA: [], fgB: [] }),
  setReactantTypes: (types) => set({ reactantTypes: types }),
  setFgA: (fgs) => set({ fgA: fgs }),
  setFgB: (fgs) => set({ fgB: fgs }),
  setExcludeCui: (val) => set({ excludeCui: val }),
  setExcludeScaleup: (val) => set({ excludeScaleup: val }),
  setIncludeNullCategories: (val) => set({ includeNullCategories: val }),
  setMinEln: (val) => set({ minEln: val }),
  setTopnZscore: (val) => set({ topnZscore: val }),
  setMaxComponents: (val) => set({ maxComponents: val }),
  setActiveTab: (tab) => set({ activeTab: tab }),
  togglePresentationMode: () =>
    set((s) => ({ presentationMode: !s.presentationMode })),
  toggleOptionsPanel: () =>
    set((s) => ({ optionsPanelOpen: !s.optionsPanelOpen })),

  resetFilters: () =>
    set((s) => ({
      // Keep current reaction types — clear everything else
      reactionTypes: s.reactionTypes,
      reactantTypes: [],
      fgA: [],
      fgB: [],
      excludeCui: true,
      excludeScaleup: true,
      includeNullCategories: true,
      minEln: DEFAULT_MIN_ELN,
      topnZscore: DEFAULT_TOPN_ZSCORE,
      maxComponents: DEFAULT_MAX_COMPONENTS,
      activeTab: 'boxplot',
      uploadedDataset: null,
    })),

  loadDataset: async () => {
    set({ isLoading: true, loadError: null });
    try {
      const rows = await loadDataset();
      set({ dataset: rows, isLoading: false });
    } catch (e) {
      set({
        loadError: e instanceof Error ? e.message : 'Failed to load dataset',
        isLoading: false,
      });
    }
  },

  setUploadedDataset: (rows) => set({ uploadedDataset: rows }),

  uploadCSV: (text, fileName) => {
    const REQUIRED_COLUMNS = [
      'ELN_ID', 'PLATENUMBER', 'Coordinate', 'AREA_TOTAL_REDUCED',
      'Base', 'Catalyst', 'Solvent', 'Ligand',
      'Reaction Type', 'FG A', 'FG B', 'FG_sorted', 'z-Score',
    ];

    try {
      const rows = parseCSVText(text);
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

      set({ uploadedDataset: rows, uploadError: null, uploadFileName: fileName ?? null });
    } catch {
      set({ uploadError: 'Failed to parse CSV file. Check the format and encoding.' });
    }
  },

  clearUploadError: () => set({ uploadError: null }),

  setFilters: (partial) => set(partial),
}));

export { DEFAULT_REACTION_TYPES, DEFAULT_FG_A, DEFAULT_FG_B, DEFAULT_REACTANT_TYPES };
export { DEFAULT_MIN_ELN, DEFAULT_TOPN_ZSCORE, DEFAULT_MAX_COMPONENTS };
