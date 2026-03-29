/**
 * Dropdown option computation — port of callbacks.py dropdown logic.
 *
 * Computes available options for FG A, FG B (conditioned on FG A),
 * reactant types, and reaction types from a dataset.
 */

import type { Row, DropdownIndex } from './types';
import { CATEGORY_OPTIONS } from './types';

/** Get unique reaction types from dataset. */
export function getReactionTypes(rows: Row[]): string[] {
  const types = new Set<string>();
  for (const row of rows) {
    if (row['Reaction Type']) types.add(row['Reaction Type']);
  }
  return Array.from(types).sort();
}

/** Get available reactant type columns (excluding FG A/B) for given reaction types. */
export function getReactantOptions(
  rows: Row[],
  reactionTypes: string[],
): string[] {
  let filtered = rows;
  if (reactionTypes.length > 0) {
    const typeSet = new Set(reactionTypes);
    filtered = rows.filter((row) => typeSet.has(row['Reaction Type']));
  }

  const available: string[] = [];
  for (const cat of CATEGORY_OPTIONS) {
    if (filtered.some((row) => row[cat] !== null && row[cat] !== undefined && row[cat] !== '')) {
      available.push(cat);
    }
  }
  return available;
}

/** Get all unique FG values (combined FG A + FG B) for given reaction types. */
export function getFgOptions(
  rows: Row[],
  reactionTypes: string[],
): string[] {
  let filtered = rows;
  if (reactionTypes.length > 0) {
    const typeSet = new Set(reactionTypes);
    filtered = rows.filter((row) => typeSet.has(row['Reaction Type']));
  }

  const fgs = new Set<string>();
  for (const row of filtered) {
    if (row['FG A']) fgs.add(row['FG A']);
    if (row['FG B']) fgs.add(row['FG B']);
  }
  return Array.from(fgs).sort();
}

/**
 * Get FG B options conditioned on selected FG A values.
 *
 * Port of callbacks._update_fg_b_options logic:
 * - If FG A has specific values: find all FGs that co-occur with any selected FG A
 * - If FG A is empty/All: return all FGs
 */
export function getFgBOptionsConditioned(
  rows: Row[],
  reactionTypes: string[],
  fgASelection: string[],
): string[] {
  let filtered = rows;
  if (reactionTypes.length > 0) {
    const typeSet = new Set(reactionTypes);
    filtered = rows.filter((row) => typeSet.has(row['Reaction Type']));
  }

  // If no specific FG A selection, return all FGs
  if (!fgASelection || fgASelection.length === 0) {
    return getFgOptions(rows, reactionTypes);
  }

  // Find FGs that co-occur with any selected FG A value
  const otherFgs = new Set<string>();
  for (const fgAVal of fgASelection) {
    for (const row of filtered) {
      if (row['FG A'] === fgAVal && row['FG B']) {
        otherFgs.add(row['FG B']);
      }
      if (row['FG B'] === fgAVal && row['FG A']) {
        otherFgs.add(row['FG A']);
      }
    }
  }

  return Array.from(otherFgs).sort();
}

// --- Index-based functions (instant, no row scanning) ---

/** Get sorted reaction types from pre-computed index. */
export function getReactionTypesFromIndex(index: DropdownIndex): string[] {
  return Object.keys(index).sort();
}

/** Get available reactant columns from index for given reaction types. */
export function getReactantOptionsFromIndex(
  index: DropdownIndex,
  reactionTypes: string[],
): string[] {
  const rts = reactionTypes.length > 0 ? reactionTypes : Object.keys(index);
  const available = new Set<string>();
  for (const rt of rts) {
    const entry = index[rt];
    if (entry) {
      for (const cat of entry.reactant_availability) {
        available.add(cat);
      }
    }
  }
  return CATEGORY_OPTIONS.filter((c) => available.has(c));
}

/** Get all unique FG values from index for given reaction types. */
export function getFgOptionsFromIndex(
  index: DropdownIndex,
  reactionTypes: string[],
): string[] {
  const rts = reactionTypes.length > 0 ? reactionTypes : Object.keys(index);
  const fgs = new Set<string>();
  for (const rt of rts) {
    const entry = index[rt];
    if (entry) {
      for (const fg of entry.fg_all_options) {
        fgs.add(fg);
      }
    }
  }
  return Array.from(fgs).sort();
}

/** Get FG B options conditioned on FG A from index. */
export function getFgBOptionsFromIndex(
  index: DropdownIndex,
  reactionTypes: string[],
  fgASelection: string[],
): string[] {
  if (!fgASelection || fgASelection.length === 0) {
    return getFgOptionsFromIndex(index, reactionTypes);
  }

  const rts = reactionTypes.length > 0 ? reactionTypes : Object.keys(index);
  const fgASet = new Set(fgASelection);
  const otherFgs = new Set<string>();

  for (const rt of rts) {
    const entry = index[rt];
    if (!entry) continue;
    for (const [fgKey, fgBValues] of Object.entries(entry.fg_b_conditioned)) {
      // Forward: selected FG A matches a key → add the conditioned values
      if (fgASet.has(fgKey)) {
        for (const fg of fgBValues) {
          otherFgs.add(fg);
        }
      }
      // Reverse: selected FG A appears in the values → add the key
      // (skip composite keys — their parts are already covered by single keys)
      if (!fgKey.includes('+')) {
        for (const fgBVal of fgBValues) {
          if (fgASet.has(fgBVal)) {
            otherFgs.add(fgKey);
            break;
          }
        }
      }
    }
  }

  return Array.from(otherFgs).sort();
}
