/**
 * 11-step filter chain — port of data_utils.filter_data().
 *
 * Orchestrates the individual filter steps from filterSteps.ts.
 * Returns filtered rows and optional statistics.
 */

import type { Row, FilterParams, FilterStats } from './types';

export type { FilterParams };
import {
  filterByReactionTypes,
  filterByReactantColumns,
  filterCopper,
  filterPrecomplexed,
  filterFgA,
  filterFgB,
  filterScaleupPlates,
  deduplicateBestZscore,
  filterTopNZscore,
  filterMinEln,
  filterMaxComponents,
} from './filterSteps';

/** Count unique ELN_IDs in a row array. */
export function countElns(rows: Row[]): number {
  const elns = new Set<string>();
  for (const row of rows) {
    if (row.ELN_ID) elns.add(row.ELN_ID);
  }
  return elns.size;
}

/** Count unique combinations of the given columns. */
function countUniqueCombos(rows: Row[], cols: string[]): number {
  const combos = new Set<string>();
  for (const row of rows) {
    const key = cols.map((col) => row[col] ?? '').join('|');
    combos.add(key);
  }
  return combos.size;
}

export interface FilterResult {
  rows: Row[];
  stats: FilterStats;
}

/** Rows surviving each of the base filters, for callers that report on them. */
export interface BaseFilterResult {
  rows: Row[];
  afterReactionTypes: Row[];
  afterReactants: Row[];
  afterFgA: Row[];
  fgAList: string[];
  fgBList: string[];
}

/**
 * Run the base filters — chain steps 1-5, everything that narrows by what the
 * user selected rather than by z-Score ranking.
 *
 * Split out so the dropdown ELN counts scope themselves through the same
 * definition the chain uses; a step added here reaches both.
 */
export function baseFilter(
  sourceRows: Row[],
  params: FilterParams,
): BaseFilterResult {
  // Step 1: Reaction types
  const afterReactionTypes = filterByReactionTypes(
    sourceRows,
    params.reactionTypes,
  );

  // Step 2: Reactant columns populated
  const afterReactants = filterByReactantColumns(
    afterReactionTypes,
    params.reactantTypes,
    params.includeNullCategories,
  );

  // Step 3: Copper catalyst filter
  let rows = filterCopper(afterReactants, params.copperFilter);

  // Step 3b: Precomplexed catalyst filter
  rows = filterPrecomplexed(rows, params.precomplexedFilter);

  // Step 4: Functional Group A
  const [afterFgA, fgAList] = filterFgA(rows, params.fgA);

  // Step 5: Functional Group B
  const [afterFgB, fgBList] = filterFgB(afterFgA, params.fgB, fgAList);

  return {
    rows: afterFgB,
    afterReactionTypes,
    afterReactants,
    afterFgA,
    fgAList,
    fgBList,
  };
}

/**
 * Run the 10-step filter chain.
 *
 * This is the core data transformation pipeline. Each step narrows the
 * dataset based on user-selected filter parameters.
 */
export function filterData(
  sourceRows: Row[],
  params: FilterParams,
): FilterResult {
  const stats: FilterStats = {};

  const base = baseFilter(sourceRows, params);
  let rows = base.rows;

  stats.wholeDataset = { elns: countElns(base.afterReactionTypes) };
  if (params.reactantTypes.length > 0) {
    stats.afterReactantFilters = { elns: countElns(base.afterReactants) };
  }
  if (base.fgAList.length > 0) {
    stats.afterFgA = { elns: countElns(base.afterFgA) };
  }
  if (base.fgBList.length > 0) {
    stats.afterFgB = { elns: countElns(rows) };
  }

  // If no rows survive the base filters (steps 1–5), no options can help
  if (rows.length === 0) {
    stats.noDataHint = 'No data exists for the selected reaction type, functional group, and reactant type combination.';
    return { rows, stats };
  }

  // Step 6: Scale-up plates
  const rowsBeforeScaleup = rows.length;
  rows = filterScaleupPlates(rows, params.excludeScaleup);
  if (rows.length === 0 && rowsBeforeScaleup > 0) {
    stats.noDataHint =
      'All data was removed by scale-up plate exclusion. Try unchecking "Exclude Scale-Up Plates" in Options.';
    return { rows, stats };
  }

  // Step 7: Deduplication
  rows = deduplicateBestZscore(rows);

  // Step 8: Top-N z-scores
  const rowsBeforeTopN = rows.length;
  rows = filterTopNZscore(
    rows,
    params.topnZscore,
    params.reactantTypes,
    params.includeNullCategories,
  );
  if (rows.length === 0 && rowsBeforeTopN > 0) {
    stats.noDataHint =
      `Try increasing "Top-N z-Score" (currently ${params.topnZscore}) in Options.`;
    return { rows, stats };
  }

  // Step 9: Min ELN count
  const rowsBeforeMinEln = rows.length;
  rows = filterMinEln(
    rows,
    params.minEln,
    params.reactantTypes,
    params.includeNullCategories,
  );
  if (rows.length === 0 && rowsBeforeMinEln > 0) {
    stats.noDataHint =
      `Try lowering "Minimum Number of ELNs" (currently ${params.minEln}) in Options.`;
    return { rows, stats };
  }

  // Compute max-components cap for the slider (before step 10)
  if (params.reactantTypes.length > 0) {
    const keyCols = params.reactantTypes.filter((rt) => rt && rt.trim());
    stats.maxComponentsCap = countUniqueCombos(rows, keyCols);
  }

  // Step 10: Max components
  const rowsBeforeMaxComp = rows.length;
  rows = filterMaxComponents(
    rows,
    params.maxComponents,
    params.reactantTypes,
    params.includeNullCategories,
  );
  if (rows.length === 0 && rowsBeforeMaxComp > 0) {
    stats.noDataHint =
      'Try increasing "Max Components to Display" in Options.';
  }

  return { rows, stats };
}
