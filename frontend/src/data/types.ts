/**
 * Core data types for the zScore-App.
 *
 * The Row interface mirrors the CSV columns from the dataset.
 * FilterParams captures all 10 filter chain parameters.
 */

/** A single row from the z-Score dataset CSV. */
export interface Row {
  ELN_ID: string;
  PLATENUMBER: string;
  Coordinate: string;
  AREA_TOTAL_REDUCED: number | null;
  Additive: string | null;
  Base: string | null;
  Catalyst: string | null;
  'Coupling Reagent': string | null;
  Solvent: string | null;
  Ligand: string | null;
  'Secondary Solvent': string | null;
  'Reaction Type': string;
  'FG A': string | null;
  'FG B': string | null;
  FG_sorted: string | null;
  FG_PAIR_SORTED: string | null;
  'z-Score': number | null;
  // Additional columns present in CSV but not used in filtering:
  [key: string]: string | number | null | undefined;
}

/** Parameters for the 10-step filter chain. */
export interface FilterParams {
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
}

/** Statistics returned alongside filtered data. */
export interface FilterStats {
  wholeDataset?: { elns: number };
  afterReactantFilters?: { elns: number };
  afterFgA?: { elns: number };
  afterFgB?: { elns: number };
  maxComponentsCap?: number;
  /** Diagnostic hint explaining why filtered rows are empty. */
  noDataHint?: string;
}

/** Default filter values — single source of truth. */
export const DEFAULTS: FilterParams = {
  reactionTypes: [],
  reactantTypes: [],
  fgA: [],
  fgB: [],
  excludeCui: true,
  excludeScaleup: true,
  includeNullCategories: true,
  minEln: 5,
  topnZscore: 5,
  maxComponents: 10,
};

/** The 13 columns required in uploaded datasets. */
export const REQUIRED_COLUMNS = [
  'ELN_ID',
  'PLATENUMBER',
  'Coordinate',
  'AREA_TOTAL_REDUCED',
  'Base',
  'Catalyst',
  'Solvent',
  'Ligand',
  'Reaction Type',
  'FG A',
  'FG B',
  'FG_sorted',
  'z-Score',
] as const;

/** Reactant category columns (excluding FG A/FG B). */
export const CATEGORY_OPTIONS = [
  'Additive',
  'Base',
  'Catalyst',
  'Coupling Reagent',
  'Solvent',
  'Ligand',
  'Secondary Solvent',
] as const;

/** Pre-computed dropdown index for instant startup. */
export interface DropdownIndexEntry {
  reactant_availability: string[];
  fg_all_options: string[];
  fg_b_conditioned: Record<string, string[]>;
}

/** Maps reaction type → pre-computed dropdown data. */
export type DropdownIndex = Record<string, DropdownIndexEntry>;

/** Analysis view tab identifiers. */
export type TabId = 'boxplot' | 'violin' | 'heatmap' | 'stats';

/** Which filter dropdown is in split mode (null = combined). */
export type SplitSelector = 'reactionTypes' | 'fgA' | 'fgB' | 'reactantTypes';

/** Metadata for a single built-in dataset version. */
export interface VersionInfo {
  id: string;
  parquet: string;
  index: string;
  label: string;
  date?: string;
}

/** Manifest listing all available dataset versions. */
export interface VersionsManifest {
  versions: VersionInfo[];
  latest: string;
}

/** Upload mode: replace built-in data or combine with it. */
export type UploadMode = 'replace' | 'combine';

/** Rank change info for a single category when comparing dataset versions. */
export interface RankDelta {
  rankChange: number;   // positive = moved up, negative = moved down
  medianDelta: number;  // change in median z-Score
  isNew: boolean;       // exists in current version but not in comparison version
}

/** URL abbreviations for split selectors. */
export const SPLIT_URL_KEYS: Record<SplitSelector, string> = {
  reactionTypes: 'rt',
  fgA: 'fga',
  fgB: 'fgb',
  reactantTypes: 'cat',
};

/** One panel of split (or combined) filtered data. */
export interface SplitPanel {
  label: string;
  rows: Row[];
  stats: FilterStats;
  reactantTypes: string[];
}

/** Reagent columns used in deduplication and scale-up detection. */
export const REAGENT_COLS = [
  'Additive',
  'Base',
  'Catalyst',
  'Coupling Reagent',
  'Solvent',
  'Ligand',
  'Secondary Solvent',
  'Tertiary Solvent',
] as const;
