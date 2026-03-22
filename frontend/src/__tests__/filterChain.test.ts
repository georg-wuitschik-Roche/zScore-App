/**
 * Filter chain tests.
 *
 * Loads the real CSV and validates basic filter behaviour.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { describe, it, expect, beforeAll } from 'vitest';
import { parseCSVText } from '../data/loader';
import { filterData } from '../data/filterChain';
import type { Row } from '../data/types';
import { DEFAULTS } from '../data/types';

// ---------------------------------------------------------------------------
// Load dataset once
// ---------------------------------------------------------------------------

let dataset: Row[];

beforeAll(async () => {
  const csvPath = resolve(__dirname, '../../public/data/z-score-peaks.csv');
  const csvText = readFileSync(csvPath, 'utf-8');
  dataset = await parseCSVText(csvText);
});

// ---------------------------------------------------------------------------
// Basic smoke tests
// ---------------------------------------------------------------------------

describe('Filter chain basics', () => {
  it('loads dataset with correct row count', () => {
    // The CSV has 67201 data rows (header excluded)
    expect(dataset.length).toBeGreaterThan(60000);
  });

  it('default filter produces non-empty result', () => {
    const { rows } = filterData(dataset, DEFAULTS);
    expect(rows.length).toBeGreaterThan(0);
  });

  it('nonexistent reaction type produces empty result', () => {
    const params = { ...DEFAULTS, reactionTypes: ['NONEXISTENT'] };
    const { rows } = filterData(dataset, params);
    expect(rows.length).toBe(0);
  });

  it('CuI exclusion removes CuI rows', () => {
    const params = {
      ...DEFAULTS,
      excludeCui: true,
      fgA: [],
      fgB: [],
    };
    const { rows } = filterData(dataset, params);
    const cuiRows = rows.filter((r) => r.Catalyst === 'CuI');
    expect(cuiRows.length).toBe(0);
  });

  it('stats contain expected keys', () => {
    const params = {
      ...DEFAULTS,
      fgA: ['RNH2', 'RNH2 a-branch'],
      fgB: ['ArBr', 'ArCl'],
    };
    const { stats } = filterData(dataset, params);
    expect(stats.wholeDataset).toBeDefined();
    expect(stats.afterFgA).toBeDefined();
    expect(stats.afterFgB).toBeDefined();
    expect(stats.maxComponentsCap).toBeDefined();
  });
});
