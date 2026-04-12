/**
 * Filter chain tests.
 *
 * Loads the real CSV and validates basic filter behaviour.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { describe, it, expect, beforeAll } from 'vitest';
import { parseDataset } from '../data/loader';
import { filterData } from '../data/filterChain';
import type { Row } from '../data/types';
import { DEFAULTS } from '../data/types';
import { isCopperCatalyst } from '../data/filterSteps';

// ---------------------------------------------------------------------------
// Load dataset once (from Parquet)
// ---------------------------------------------------------------------------

let dataset: Row[];

beforeAll(async () => {
  const parquetPath = resolve(__dirname, '../../public/data/z-score-peaks.parquet');
  const buffer = readFileSync(parquetPath);
  dataset = await parseDataset(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
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
    const params = {
      ...DEFAULTS,
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
    };
    const { rows } = filterData(dataset, params);
    expect(rows.length).toBeGreaterThan(0);
  });

  it('nonexistent reaction type produces empty result', () => {
    const params = { ...DEFAULTS, reactionTypes: ['NONEXISTENT'] };
    const { rows } = filterData(dataset, params);
    expect(rows.length).toBe(0);
  });

  it('copper exclusion removes all copper catalyst rows', () => {
    const params = {
      ...DEFAULTS,
      reactionTypes: ['Buchwald-Hartwig'],
      copperFilter: 'exclude' as const,
    };
    const { rows } = filterData(dataset, params);
    const copperRows = rows.filter((r) => isCopperCatalyst(r.Catalyst as string | null));
    expect(copperRows.length).toBe(0);
  });

  it('stats contain expected keys', () => {
    const params = {
      ...DEFAULTS,
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
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
