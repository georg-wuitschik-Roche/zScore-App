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
}

/** Default filter values — single source of truth. */
export const DEFAULTS: FilterParams = {
  reactionTypes: ['Buchwald-Hartwig'],
  reactantTypes: ['Catalyst'],
  fgA: ['RNH2 a-branch', 'RNH2'],
  fgB: ['ArBr', 'ArCl'],
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
