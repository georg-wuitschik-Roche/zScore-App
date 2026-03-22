/**
 * Dropdown option computation — port of callbacks.py dropdown logic.
 *
 * Computes available options for FG A, FG B (conditioned on FG A),
 * reactant types, and reaction types from a dataset.
 */

import type { Row } from './types';
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
