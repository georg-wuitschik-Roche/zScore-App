/**
 * Golden fixture tests for stats table (descriptive statistics).
 *
 * Validates that the TypeScript filter chain produces the same row counts,
 * ELN counts, and z-Score / AREA_TOTAL_REDUCED descriptive statistics
 * as the Python reference implementation.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { describe, it, expect, beforeAll } from 'vitest';
import { parseDataset } from '../data/loader';
import { filterData } from '../data/filterChain';
import type { Row, FilterParams } from '../data/types';

// ---------------------------------------------------------------------------
// Golden fixture types
// ---------------------------------------------------------------------------

interface ColumnStats {
  count: number;
  mean: number;
  std: number;
  min: number;
  '25%': number;
  '50%': number;
  '75%': number;
  max: number;
}

interface StatsGoldenEntry {
  params: Record<string, unknown>;
  row_count: number;
  eln_count: number;
  columns: Record<string, ColumnStats>;
}

type StatsGolden = Record<string, StatsGoldenEntry>;

// ---------------------------------------------------------------------------
// Load dataset and golden fixtures
// ---------------------------------------------------------------------------

let dataset: Row[];

const goldenDir = resolve(__dirname, '../../golden');
const golden: StatsGolden = JSON.parse(
  readFileSync(resolve(goldenDir, 'stats_table.json'), 'utf-8'),
);

beforeAll(async () => {
  const parquetPath = resolve(__dirname, '../../public/data/z-score-peaks.parquet');
  const buffer = readFileSync(parquetPath);
  dataset = await parseDataset(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Convert Python golden fixture params to TypeScript FilterParams.
 *
 * Python param format:
 *   exclude_cui: ["exclude_cui"] | null/absent  -> excludeCui: boolean
 *   exclude_scaleup: [true] | null/absent       -> excludeScaleup: boolean
 *   include_null_categories: [true] | null/absent -> includeNullCategories: boolean
 *   reaction_types: string[] | absent            -> reactionTypes: string[]
 *   reactant_types: string[]                     -> reactantTypes: string[]
 *   fg_a: string[] | absent                      -> fgA: string[]
 *   fg_b: string[] | absent                      -> fgB: string[]
 *   topn_zscore: number | absent                 -> topnZscore: number
 *   min_eln: number | absent                     -> minEln: number
 *   max_components: number | null/absent         -> maxComponents: number
 */
function convertParams(pyParams: Record<string, unknown>): FilterParams {
  return {
    reactionTypes: (pyParams.reaction_types as string[]) ?? [],
    reactantTypes: (pyParams.reactant_types as string[]) ?? [],
    fgA: (pyParams.fg_a as string[]) ?? [],
    fgB: (pyParams.fg_b as string[]) ?? [],
    excludeCui: Array.isArray(pyParams.exclude_cui)
      ? pyParams.exclude_cui.includes('exclude_cui')
      : false,
    excludeScaleup: Array.isArray(pyParams.exclude_scaleup)
      ? pyParams.exclude_scaleup.length > 0
      : false,
    includeNullCategories: Array.isArray(pyParams.include_null_categories)
      ? pyParams.include_null_categories.length > 0
      : false,
    minEln: (pyParams.min_eln as number) ?? 0,
    topnZscore: (pyParams.topn_zscore as number) ?? 0,
    maxComponents: (pyParams.max_components as number) ?? 0,
  };
}

/** Linear interpolation percentile (matches numpy/pandas default). */
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Compute mean of a numeric array. */
function mean(values: number[]): number {
  if (values.length === 0) return NaN;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

/** Compute sample standard deviation (ddof=1, matching pandas default). */
function std(values: number[]): number {
  if (values.length <= 1) return NaN;
  const m = mean(values);
  let sumSq = 0;
  for (const v of values) sumSq += (v - m) * (v - m);
  return Math.sqrt(sumSq / (values.length - 1));
}

/**
 * Compute descriptive statistics for a numeric column.
 * Matches pandas DataFrame.describe() output.
 */
function computeStats(values: number[]): ColumnStats {
  const sorted = values.slice().sort((a, b) => a - b);
  return {
    count: values.length,
    mean: mean(values),
    std: std(values),
    min: sorted[0],
    '25%': percentile(sorted, 25),
    '50%': percentile(sorted, 50),
    '75%': percentile(sorted, 75),
    max: sorted[sorted.length - 1],
  };
}

/** Extract valid numeric values for a column from filtered rows. */
function extractNumericColumn(rows: Row[], col: string): number[] {
  const values: number[] = [];
  for (const row of rows) {
    const v = row[col];
    if (typeof v === 'number' && !isNaN(v)) {
      values.push(v);
    }
  }
  return values;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Stats table (golden fixtures)', () => {
  const testKeys = Object.keys(golden);

  it(`has ${testKeys.length} test cases in golden file`, () => {
    expect(testKeys.length).toBeGreaterThan(0);
  });

  // Lazy caches: filterData called once per test key, stats computed once per column
  const rowsCache = new Map<string, Row[]>();
  function getRows(key: string, params: FilterParams): Row[] {
    let rows = rowsCache.get(key);
    if (!rows) {
      rows = filterData(dataset, params).rows;
      rowsCache.set(key, rows);
    }
    return rows;
  }

  const statsCache = new Map<string, ColumnStats>();
  function getColStats(key: string, params: FilterParams, col: string): ColumnStats {
    const cacheKey = `${key}|${col}`;
    let stats = statsCache.get(cacheKey);
    if (!stats) {
      const values = extractNumericColumn(getRows(key, params), col);
      stats = computeStats(values);
      statsCache.set(cacheKey, stats);
    }
    return stats;
  }

  for (const testKey of testKeys) {
    const entry = golden[testKey];
    const params = convertParams(entry.params);

    describe(`${testKey}`, () => {
      it('row count matches', () => {
        expect(getRows(testKey, params).length).toBe(entry.row_count);
      });

      it('unique ELN count matches', () => {
        const rows = getRows(testKey, params);
        const elns = new Set<string>();
        for (const row of rows) {
          if (row.ELN_ID) elns.add(row.ELN_ID);
        }
        expect(elns.size).toBe(entry.eln_count);
      });

      // Test each column's descriptive statistics
      for (const [colName, expectedStats] of Object.entries(entry.columns)) {
        describe(`column: ${colName}`, () => {
          it('count matches', () => {
            expect(getColStats(testKey, params, colName).count).toBe(expectedStats.count);
          });

          it('mean matches within tolerance', () => {
            expect(Math.abs(getColStats(testKey, params, colName).mean - expectedStats.mean)).toBeLessThan(1e-2);
          });

          it('std matches within tolerance', () => {
            expect(Math.abs(getColStats(testKey, params, colName).std - expectedStats.std)).toBeLessThan(1e-2);
          });

          it('min matches', () => {
            expect(Math.abs(getColStats(testKey, params, colName).min - expectedStats.min)).toBeLessThan(1e-4);
          });

          it('25th percentile matches', () => {
            expect(Math.abs(getColStats(testKey, params, colName)['25%'] - expectedStats['25%'])).toBeLessThan(1e-4);
          });

          it('50th percentile (median) matches', () => {
            expect(Math.abs(getColStats(testKey, params, colName)['50%'] - expectedStats['50%'])).toBeLessThan(1e-4);
          });

          it('75th percentile matches', () => {
            expect(Math.abs(getColStats(testKey, params, colName)['75%'] - expectedStats['75%'])).toBeLessThan(1e-4);
          });

          it('max matches', () => {
            expect(Math.abs(getColStats(testKey, params, colName).max - expectedStats.max)).toBeLessThan(1e-4);
          });
        });
      }
    });
  }
});
