/**
 * Golden fixture tests for heatmap pivot values.
 *
 * Validates that the TypeScript filter chain + median pivot computation
 * produces the same 2D matrix, axis categories, and ELN counts
 * as the Python reference implementation.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { describe, it, expect, beforeAll } from 'vitest';
import { parseCSVText } from '../data/loader';
import { filterData } from '../data/filterChain';
import { median } from '../data/filterSteps';
import type { Row, FilterParams } from '../data/types';

// ---------------------------------------------------------------------------
// Golden fixture types
// ---------------------------------------------------------------------------

interface HeatmapGoldenEntry {
  row_count: number;
  y_order: string[];
  x_order: string[];
  n_y: number;
  n_x: number;
  cell_medians: Record<string, number | null>;
  cell_eln_counts: Record<string, number>;
}

type HeatmapGolden = Record<string, HeatmapGoldenEntry>;

// ---------------------------------------------------------------------------
// Load dataset and golden fixtures
// ---------------------------------------------------------------------------

let dataset: Row[];

const goldenDir = resolve(__dirname, '../../golden');
const golden: HeatmapGolden = JSON.parse(
  readFileSync(resolve(goldenDir, 'heatmap_pivots.json'), 'utf-8'),
);

beforeAll(() => {
  const csvPath = resolve(__dirname, '../../public/data/z-score-peaks.csv');
  dataset = parseCSVText(readFileSync(csvPath, 'utf-8'));
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Get a string column value from a row, treating null/undefined/empty as null. */
function getVal(row: Row, col: string): string | null {
  const v = row[col];
  if (v === null || v === undefined || v === '') return null;
  return String(v);
}

/**
 * Build FilterParams matching the Python golden generator settings.
 *
 * The generator uses: excludeCui, excludeScaleup, includeNull, minEln=3, topn=3,
 * no FG filtering, no maxComponents cap.
 */
function buildHeatmapParams(
  reactionType: string,
  reactantTypes: string[],
): FilterParams {
  return {
    reactionTypes: [reactionType],
    reactantTypes,
    fgA: [],
    fgB: [],
    excludeCui: true,
    excludeScaleup: true,
    includeNullCategories: true,
    minEln: 3,
    topnZscore: 3,
    maxComponents: 0,
  };
}

/**
 * Compute the pivot median for a specific (yVal, xVal) cell from filtered rows.
 * Returns null if no valid z-Scores exist for that cell.
 */
function computeCellMedian(
  rows: Row[],
  yCol: string,
  xCol: string,
  yVal: string,
  xVal: string,
): number | null {
  const scores: number[] = [];
  for (const row of rows) {
    if (getVal(row, yCol) === yVal && getVal(row, xCol) === xVal) {
      const z = row['z-Score'];
      if (z !== null && z !== undefined && !isNaN(z)) {
        scores.push(z);
      }
    }
  }
  if (scores.length === 0) return null;
  return median(scores);
}

/**
 * Count unique ELN_IDs for a specific (yVal, xVal) cell from filtered rows.
 */
function computeCellElnCount(
  rows: Row[],
  yCol: string,
  xCol: string,
  yVal: string,
  xVal: string,
): number {
  const elns = new Set<string>();
  for (const row of rows) {
    if (getVal(row, yCol) === yVal && getVal(row, xCol) === xVal) {
      if (row.ELN_ID) elns.add(row.ELN_ID);
    }
  }
  return elns.size;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Heatmap pivots (golden fixtures)', () => {
  const pivotKeys = Object.keys(golden);

  it(`has ${pivotKeys.length} pivot entries in golden file`, () => {
    expect(pivotKeys.length).toBeGreaterThan(0);
  });

  for (const pivotKey of pivotKeys) {
    const expected = golden[pivotKey];
    // Key format: "ReactionType|ReactantType1|ReactantType2[|ReactantType3]"
    const parts = pivotKey.split('|');
    const reactionType = parts[0];
    const reactantTypes = parts.slice(1);
    // y-axis = first reactant type, x-axis = second reactant type
    const yCol = reactantTypes[0];
    const xCol = reactantTypes[1];

    describe(`${pivotKey}`, () => {
      it('row count matches', () => {
        const params = buildHeatmapParams(reactionType, reactantTypes);
        const { rows } = filterData(dataset, params);
        expect(rows.length).toBe(expected.row_count);
      });

      it('y-axis categories match (as set)', () => {
        const params = buildHeatmapParams(reactionType, reactantTypes);
        const { rows } = filterData(dataset, params);

        const yCategories = new Set<string>();
        for (const row of rows) {
          const v = getVal(row, yCol);
          if (v !== null) yCategories.add(v);
        }

        expect(Array.from(yCategories).sort()).toEqual(
          expected.y_order.slice().sort(),
        );
      });

      it('x-axis categories match (as set)', () => {
        const params = buildHeatmapParams(reactionType, reactantTypes);
        const { rows } = filterData(dataset, params);

        const xCategories = new Set<string>();
        for (const row of rows) {
          const v = getVal(row, xCol);
          if (v !== null) xCategories.add(v);
        }

        expect(Array.from(xCategories).sort()).toEqual(
          expected.x_order.slice().sort(),
        );
      });

      it('category counts match', () => {
        expect(expected.y_order.length).toBe(expected.n_y);
        expect(expected.x_order.length).toBe(expected.n_x);
      });

      it('cell medians match within tolerance', () => {
        const params = buildHeatmapParams(reactionType, reactantTypes);
        const { rows } = filterData(dataset, params);

        for (const [cellKey, expectedMedian] of Object.entries(
          expected.cell_medians,
        )) {
          const [yVal, xVal] = cellKey.split('|');
          const actual = computeCellMedian(rows, yCol, xCol, yVal, xVal);

          if (expectedMedian === null) {
            expect(actual).toBeNull();
          } else {
            expect(actual).not.toBeNull();
            expect(Math.abs(actual! - expectedMedian)).toBeLessThan(1e-3);
          }
        }
      });

      it('cell ELN counts match', () => {
        const params = buildHeatmapParams(reactionType, reactantTypes);
        const { rows } = filterData(dataset, params);

        for (const [cellKey, expectedCount] of Object.entries(
          expected.cell_eln_counts,
        )) {
          const [yVal, xVal] = cellKey.split('|');
          const actual = computeCellElnCount(rows, yCol, xCol, yVal, xVal);
          expect(actual).toBe(expectedCount);
        }
      });
    });
  }
});
