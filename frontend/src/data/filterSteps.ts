/**
 * Individual filter step functions — port of data_utils.py filter chain.
 *
 * Each function is a pure transformation: Row[] → Row[].
 * The 10-step chain is orchestrated by filterChain.ts.
 */

import type { Row, CopperFilter } from './types';
import { REAGENT_COLS } from './types';

const NAN_SENTINEL = '__NAN__';
const NULL_SENTINEL = '__NULL_CATEGORY__';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Normalize FG input: filter out 'All', handle null/empty. */
export function normalizeFgInput(
  fgInput: string | string[] | null | undefined,
): string[] {
  if (!fgInput) return [];
  if (typeof fgInput === 'string') {
    return fgInput !== 'All' ? [fgInput] : [];
  }
  if (Array.isArray(fgInput)) {
    return fgInput.filter((fg) => fg !== 'All');
  }
  return [];
}

/** Get a row's column value, treating null/undefined/empty as null. */
function getVal(row: Row, col: string): string | null {
  const v = row[col];
  if (v === null || v === undefined || v === '') return null;
  return String(v);
}

/** Compute median of a numeric array. Returns NaN for empty arrays. */
export function median(nums: number[]): number {
  if (nums.length === 0) return NaN;
  const sorted = nums.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// ---------------------------------------------------------------------------
// Step 1: Filter by Reaction Types
// ---------------------------------------------------------------------------

export function filterByReactionTypes(
  rows: Row[],
  reactionTypes: string[],
): Row[] {
  if (!reactionTypes || reactionTypes.length === 0) return rows;
  const typeSet = new Set(reactionTypes);
  return rows.filter((row) => typeSet.has(row['Reaction Type']));
}

// ---------------------------------------------------------------------------
// Step 2: Filter by Reactant Columns Populated
// ---------------------------------------------------------------------------

export function filterByReactantColumns(
  rows: Row[],
  reactantTypes: string[],
  includeNull: boolean,
): Row[] {
  if (!reactantTypes || reactantTypes.length === 0 || includeNull) return rows;
  return rows.filter((row) => {
    for (const rt of reactantTypes) {
      if (rt && getVal(row, rt) === null) return false;
    }
    return true;
  });
}

// ---------------------------------------------------------------------------
// Step 3: Copper catalyst filter
// ---------------------------------------------------------------------------

const COPPER_RE = /cu|copper/i;

export function isCopperCatalyst(catalyst: string | null): boolean {
  return catalyst !== null && COPPER_RE.test(catalyst);
}

export function filterCopper(rows: Row[], mode: CopperFilter): Row[] {
  if (mode === 'include') return rows;
  return rows.filter((row) => {
    const cat = getVal(row, 'Catalyst');
    const isCopper = isCopperCatalyst(cat);
    return mode === 'exclude' ? !isCopper : isCopper;
  });
}

// ---------------------------------------------------------------------------
// Step 4: Filter by Functional Group A
// ---------------------------------------------------------------------------

export function filterFgA(
  rows: Row[],
  fgA: string | string[] | null,
): [Row[], string[]] {
  const fgAList = normalizeFgInput(fgA);
  if (fgAList.length === 0) return [rows, []];

  const fgSet = new Set(fgAList);
  const filtered = rows.filter(
    (row) => fgSet.has(row['FG A'] ?? '') || fgSet.has(row['FG B'] ?? ''),
  );
  return [filtered, fgAList];
}

// ---------------------------------------------------------------------------
// Step 5: Filter by Functional Group B (considering FG A pairs)
// ---------------------------------------------------------------------------

export function filterFgB(
  rows: Row[],
  fgB: string | string[] | null,
  fgAList: string[],
): [Row[], string[]] {
  const fgBList = normalizeFgInput(fgB);
  if (fgBList.length === 0) return [rows, fgBList];

  if (fgAList.length > 0) {
    // Both specified: match sorted pairs
    const pairSet = new Set<string>();
    for (const fa of fgAList) {
      for (const fb of fgBList) {
        pairSet.add([fa, fb].sort().join(', '));
      }
    }
    const filtered = rows.filter((row) =>
      pairSet.has(row['FG_PAIR_SORTED'] ?? ''),
    );
    return [filtered, fgBList];
  } else {
    // Only FG B specified: match in either column
    const fgSet = new Set(fgBList);
    const filtered = rows.filter(
      (row) => fgSet.has(row['FG A'] ?? '') || fgSet.has(row['FG B'] ?? ''),
    );
    return [filtered, fgBList];
  }
}

// ---------------------------------------------------------------------------
// Step 6: Filter Scale-up Plates
// ---------------------------------------------------------------------------

export function filterScaleupPlates(
  rows: Row[],
  excludeScaleup: boolean,
): Row[] {
  if (!excludeScaleup) return rows;

  const reagentCols = REAGENT_COLS.filter(
    (col) => rows.length > 0 && col in rows[0],
  );
  if (reagentCols.length === 0) return rows;

  // Group by (ELN_ID, PLATENUMBER), track unique reagent values
  const plateReagents = new Map<string, Set<string>[]>();

  for (const row of rows) {
    const key = `${row.ELN_ID}|${row.PLATENUMBER}`;
    if (!plateReagents.has(key)) {
      plateReagents.set(
        key,
        reagentCols.map(() => new Set()),
      );
    }
    const sets = plateReagents.get(key)!;
    for (let i = 0; i < reagentCols.length; i++) {
      const val = getVal(row, reagentCols[i]);
      if (val !== null) sets[i].add(val);
    }
  }

  // Keep plates where any reagent has >1 unique value
  const keepPlates = new Set<string>();
  for (const [key, sets] of plateReagents) {
    if (sets.some((s) => s.size > 1)) keepPlates.add(key);
  }

  return rows.filter((row) =>
    keepPlates.has(`${row.ELN_ID}|${row.PLATENUMBER}`),
  );
}

// ---------------------------------------------------------------------------
// Step 7: Deduplication — keep best z-Score per reagent combination
// ---------------------------------------------------------------------------

export function deduplicateBestZscore(rows: Row[]): Row[] {
  const dedupCols = [
    'ELN_ID',
    ...REAGENT_COLS.filter((col) => rows.length > 0 && col in rows[0]),
  ];

  // Group by composite key (NaN → sentinel)
  const groups = new Map<string, number[]>();
  for (let i = 0; i < rows.length; i++) {
    const keyParts = dedupCols.map((col) => getVal(rows[i], col) ?? NAN_SENTINEL);
    const key = keyParts.join('|');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(i);
  }

  // For each group, keep the row with highest z-Score.
  // Ties broken by original row index (lowest index wins) — matches
  // Python's rank(method='first', ascending=False).
  // Groups where ALL z-Scores are NaN are dropped (matches pandas rank() NaN behavior).
  const keepIndices = new Set<number>();
  for (const indices of groups.values()) {
    // Filter to rows with valid z-Score
    const validIndices = indices.filter((i) => {
      const z = rows[i]['z-Score'];
      return z !== null && z !== undefined && !isNaN(z);
    });
    if (validIndices.length === 0) continue; // skip all-NaN groups
    validIndices.sort((a, b) => {
      const diff =
        (rows[b]['z-Score'] as number) - (rows[a]['z-Score'] as number);
      return diff !== 0 ? diff : a - b; // stable: lower index wins tie
    });
    keepIndices.add(validIndices[0]);
  }

  return rows.filter((_, i) => keepIndices.has(i));
}

// ---------------------------------------------------------------------------
// Step 8: Top-N z-Scores per ELN + reactant combination
// ---------------------------------------------------------------------------

export function filterTopNZscore(
  rows: Row[],
  topN: number,
  reactantTypes: string[],
  includeNull: boolean,
): Row[] {
  if (!topN || topN <= 0 || !reactantTypes || reactantTypes.length === 0)
    return rows;

  const rankCols = ['ELN_ID', ...reactantTypes].filter(
    (col) => rows.length > 0 && col in rows[0],
  );
  if (rankCols.length < 2) return rows;

  // Group by rank columns
  const groups = new Map<string, number[]>();
  for (let i = 0; i < rows.length; i++) {
    const keyParts = rankCols.map((col) => {
      const val = getVal(rows[i], col);
      return val === null && includeNull ? NULL_SENTINEL : (val ?? NULL_SENTINEL);
    });
    const key = keyParts.join('|');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(i);
  }

  // For each group, keep top N by z-Score
  const keepIndices = new Set<number>();
  for (const indices of groups.values()) {
    const sorted = [...indices].sort(
      (a, b) =>
        (rows[b]['z-Score'] ?? -Infinity) - (rows[a]['z-Score'] ?? -Infinity),
    );
    for (let i = 0; i < Math.min(topN, sorted.length); i++) {
      keepIndices.add(sorted[i]);
    }
  }

  return rows.filter((_, i) => keepIndices.has(i));
}

// ---------------------------------------------------------------------------
// Step 9: Min ELN count per category group
// ---------------------------------------------------------------------------

export function filterMinEln(
  rows: Row[],
  minEln: number,
  reactantTypes: string[],
  includeNull: boolean,
): Row[] {
  if (!minEln || minEln <= 0 || !reactantTypes || reactantTypes.length === 0)
    return rows;

  const groupCols = [
    'Reaction Type',
    ...reactantTypes.filter((rt) => rt && rt.trim()),
  ];

  // Count unique ELNs per group
  const groupElns = new Map<string, Set<string>>();
  for (const row of rows) {
    const keyParts = groupCols.map((col) => {
      const val = getVal(row, col);
      return val === null && includeNull ? NULL_SENTINEL : (val ?? NULL_SENTINEL);
    });
    const key = keyParts.join('|');
    if (!groupElns.has(key)) groupElns.set(key, new Set());
    const elnId = row.ELN_ID;
    if (elnId) groupElns.get(key)!.add(elnId);
  }

  // Keep rows where group has >= minEln unique ELNs
  return rows.filter((row) => {
    const keyParts = groupCols.map((col) => {
      const val = getVal(row, col);
      return val === null && includeNull ? NULL_SENTINEL : (val ?? NULL_SENTINEL);
    });
    const key = keyParts.join('|');
    return (groupElns.get(key)?.size ?? 0) >= minEln;
  });
}

// ---------------------------------------------------------------------------
// Step 10: Max components by median z-Score
// ---------------------------------------------------------------------------

export function filterMaxComponents(
  rows: Row[],
  maxComponents: number,
  reactantTypes: string[],
  includeNull: boolean,
): Row[] {
  if (
    !maxComponents ||
    maxComponents <= 0 ||
    !reactantTypes ||
    reactantTypes.length === 0
  )
    return rows;

  const keyCols = reactantTypes.filter((rt) => rt && rt.trim());
  if (keyCols.length === 0) return rows;

  // Count unique combinations
  const uniqueCombos = new Set<string>();
  for (const row of rows) {
    const key = keyCols
      .map((col) => getVal(row, col) ?? (includeNull ? NULL_SENTINEL : ''))
      .join('|');
    uniqueCombos.add(key);
  }

  if (maxComponents >= uniqueCombos.size) return rows;

  // Compute median z-Score per combination
  const groupScores = new Map<string, number[]>();
  for (const row of rows) {
    const key = keyCols
      .map((col) => getVal(row, col) ?? (includeNull ? NULL_SENTINEL : ''))
      .join('|');
    if (!groupScores.has(key)) groupScores.set(key, []);
    const z = row['z-Score'];
    if (z !== null && z !== undefined && !isNaN(z)) {
      groupScores.get(key)!.push(z);
    }
  }

  // Sort by median descending, then alphabetically for deterministic tie-breaking.
  // This matches the golden fixture generator which uses the same convention.
  const mediansByCombo = Array.from(groupScores.entries())
    .map(([combo, scores]) => ({ combo, median: median(scores) }))
    .sort((a, b) => {
      const diff = b.median - a.median;
      if (Math.abs(diff) > 1e-9) return diff;
      return a.combo.localeCompare(b.combo);
    });

  const topCombos = new Set(
    mediansByCombo.slice(0, maxComponents).map((item) => item.combo),
  );

  return rows.filter((row) => {
    const key = keyCols
      .map((col) => getVal(row, col) ?? (includeNull ? NULL_SENTINEL : ''))
      .join('|');
    return topCombos.has(key);
  });
}
