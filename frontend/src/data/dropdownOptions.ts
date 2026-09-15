/**
 * Dropdown option computation — port of callbacks.py dropdown logic.
 *
 * Computes available options for FG A, FG B (conditioned on FG A),
 * reactant types, and reaction types from a dataset.
 *
 * The `get*ElnCounts` functions annotate each option with the number of
 * distinct ELNs behind it. Each one applies every other active filter and
 * leaves out only the dimension its own dropdown controls, so the number
 * answers "how many ELNs do I get if I pick this, on top of everything I have
 * already selected". Picking an option therefore lands on its stated count,
 * up to the late chain steps (dedup, top-N, min-ELN) which the `ELNs:` badges
 * also exclude.
 */

import type { Row, DropdownIndex, FilterParams } from './types';
import { CATEGORY_OPTIONS } from './types';
import { filterByReactionTypes } from './filterSteps';
import { baseFilter } from './filterChain';

/** The filter dimension a dropdown owns, and therefore must not apply itself. */
type CountDimension = 'reactionTypes' | 'reactantTypes' | 'fgA' | 'fgB';

/**
 * Apply the base filters with the dropdown's own dimension left out.
 *
 * Every base filter no-ops on an empty selection, so "exclude a dimension" is
 * just blanking it. Leaving it out is what makes the counts useful: applying
 * it would collapse every option to the already-selected values.
 */
function scopeForDimension(
  rows: Row[],
  params: FilterParams,
  exclude: CountDimension,
): Row[] {
  return baseFilter(rows, { ...params, [exclude]: [] }).rows;
}

/** True when a reactant column is populated on this row. */
function hasValue(row: Row, col: string): boolean {
  const value = row[col];
  return value !== null && value !== undefined && value !== '';
}

/** Record an ELN against an option, skipping null/empty on either side. */
function addEln(
  counter: Map<string, Set<string>>,
  option: string | null | undefined,
  elnId: string | null | undefined,
): void {
  if (!option || !elnId) return;
  let elns = counter.get(option);
  if (!elns) {
    elns = new Set();
    counter.set(option, elns);
  }
  elns.add(elnId);
}

/** Collapse an option -> ELN set map into option -> distinct ELN count. */
function collapse(counter: Map<string, Set<string>>): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const [option, elns] of counter) {
    counts[option] = elns.size;
  }
  return counts;
}

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
  const filtered = filterByReactionTypes(rows, reactionTypes);

  const available: string[] = [];
  for (const cat of CATEGORY_OPTIONS) {
    if (filtered.some((row) => hasValue(row, cat))) {
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
  const filtered = filterByReactionTypes(rows, reactionTypes);

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
  const filtered = filterByReactionTypes(rows, reactionTypes);

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

// --- Per-option ELN counts (mirror the option functions above) ---

/**
 * Count distinct ELNs per FG value within already-scoped rows.
 *
 * With a partner selection active, only ELNs that pair a candidate with one of
 * the partner values count — the same unordered-pair rule `filterFgB` applies.
 * With no partner selected, a candidate counts in either column, matching
 * `filterFgA`.
 */
function countFgValues(
  rows: Row[],
  partnerSelection: string[],
): Record<string, number> {
  const counter = new Map<string, Set<string>>();

  if (partnerSelection.length === 0) {
    for (const row of rows) {
      // A homo-pair row lands in the same bucket twice; the Set absorbs it.
      addEln(counter, row['FG A'], row.ELN_ID);
      addEln(counter, row['FG B'], row.ELN_ID);
    }
    return collapse(counter);
  }

  const partnerSet = new Set(partnerSelection);
  for (const row of rows) {
    const fgA = row['FG A'];
    const fgB = row['FG B'];
    if (!fgA || !fgB) continue;
    if (partnerSet.has(fgA)) addEln(counter, fgB, row.ELN_ID);
    if (partnerSet.has(fgB)) addEln(counter, fgA, row.ELN_ID);
  }
  return collapse(counter);
}

/** Count distinct ELNs per reaction type, honouring the other active filters. */
export function getReactionTypeElnCounts(
  rows: Row[],
  params: FilterParams,
): Record<string, number> {
  const scoped = scopeForDimension(rows, params, 'reactionTypes');

  const counter = new Map<string, Set<string>>();
  for (const row of scoped) {
    addEln(counter, row['Reaction Type'], row.ELN_ID);
  }
  return collapse(counter);
}

/** Count distinct ELNs per FG A option, honouring the other active filters. */
export function getFgAElnCounts(
  rows: Row[],
  params: FilterParams,
): Record<string, number> {
  return countFgValues(scopeForDimension(rows, params, 'fgA'), params.fgB);
}

/** Count distinct ELNs per FG B option, honouring the other active filters. */
export function getFgBElnCounts(
  rows: Row[],
  params: FilterParams,
): Record<string, number> {
  return countFgValues(scopeForDimension(rows, params, 'fgB'), params.fgA);
}

/**
 * Count distinct ELNs per reactant column that have that column populated,
 * honouring the other active filters.
 *
 * This is an availability count — with the default `includeNullCategories` the
 * reactant filter itself is a no-op, so the number answers whether a column
 * has enough data under the current filters to be worth grouping by.
 */
export function getReactantElnCounts(
  rows: Row[],
  params: FilterParams,
): Record<string, number> {
  const scoped = scopeForDimension(rows, params, 'reactantTypes');

  const counter = new Map<string, Set<string>>();
  for (const row of scoped) {
    for (const cat of CATEGORY_OPTIONS) {
      if (hasValue(row, cat)) addEln(counter, cat, row.ELN_ID);
    }
  }
  return collapse(counter);
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
