/**
 * Tests for version comparison: resolveComparisonVersion, computeRankDeltas,
 * and the split-by-reactantTypes filtering scenario.
 */
import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';
import { parquetRead } from 'hyparquet';
import { filterData } from '../data/filterChain';
import { resolveComparisonVersion, computeRankDeltas } from '../data/comparison';
import type { Row, FilterParams, VersionInfo } from '../data/types';

// ---------------------------------------------------------------------------
// Parquet helpers (reuse the same cleanRow logic as loader.ts)
// ---------------------------------------------------------------------------

function parseBuffer(buffer: ArrayBuffer): Promise<Record<string, unknown>[]> {
  return new Promise((resolve) => {
    parquetRead({
      file: buffer,
      rowFormat: 'object',
      onComplete: (data: Record<string, unknown>[]) => resolve(data),
    });
  });
}

function cleanRow(raw: Record<string, unknown>): Row {
  const row = { ...raw } as Record<string, unknown>;
  for (const key of Object.keys(row)) {
    if (typeof row[key] === 'bigint') row[key] = Number(row[key]);
  }
  for (const col of [
    'Additive', 'Base', 'Catalyst', 'Coupling Reagent',
    'Solvent', 'Ligand', 'Secondary Solvent', 'Tertiary Solvent',
  ]) {
    if (col in row) {
      const v = row[col];
      if (v === null || v === undefined) { row[col] = null; continue; }
      const s = String(v);
      if (s === '' || s === 'nan' || s === 'NaN') { row[col] = null; continue; }
      row[col] = s;
    }
  }
  if (!row.FG_PAIR_SORTED) {
    const fgSorted = row['FG_sorted'];
    if (fgSorted && typeof fgSorted === 'string') {
      row.FG_PAIR_SORTED = fgSorted;
    }
  }
  return row as Row;
}

async function loadParquet(filename: string): Promise<Row[]> {
  const filePath = path.join(__dirname, '../../public/data', filename);
  const buf = fs.readFileSync(filePath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const raw = await parseBuffer(ab);
  return raw.map(cleanRow);
}

// ---------------------------------------------------------------------------
// Fixture helpers for unit tests (no parquet needed)
// ---------------------------------------------------------------------------

function makeRow(overrides: Partial<Row>): Row {
  return {
    ELN_ID: 'ELN001',
    PLATENUMBER: '1',
    Coordinate: 'A1',
    AREA_TOTAL_REDUCED: 50,
    Base: null,
    Catalyst: null,
    Solvent: null,
    Ligand: null,
    Additive: null,
    'Coupling Reagent': null,
    'Secondary Solvent': null,
    'Reaction Type': 'TestRx',
    'FG A': null,
    'FG B': null,
    FG_sorted: null,
    FG_PAIR_SORTED: null,
    'z-Score': 0,
    ...overrides,
  };
}

const VERSIONS: VersionInfo[] = [
  { id: 'v1', parquet: '/data/v1.parquet', index: '/data/v1-dropdown-index.json', label: 'v1', date: '2025-09-25' },
  { id: 'v2', parquet: '/data/v2.parquet', index: '/data/v2-dropdown-index.json', label: 'v2', date: '2026-03-29' },
];

// ---------------------------------------------------------------------------
// resolveComparisonVersion
// ---------------------------------------------------------------------------

describe('resolveComparisonVersion', () => {
  it('auto-resolves to the other version (v2 active → v1)', () => {
    expect(resolveComparisonVersion(VERSIONS, 'v2', null)).toBe('v1');
  });

  it('auto-resolves to the other version (v1 active → v2)', () => {
    expect(resolveComparisonVersion(VERSIONS, 'v1', null)).toBe('v2');
  });

  it('uses explicit version when different from active', () => {
    expect(resolveComparisonVersion(VERSIONS, 'v2', 'v1')).toBe('v1');
  });

  it('ignores explicit version when same as active (self-comparison guard)', () => {
    // Should fall through to auto-resolve instead of comparing v2 against itself
    expect(resolveComparisonVersion(VERSIONS, 'v2', 'v2')).toBe('v1');
  });

  it('allows explicit == active when upload is present (compare upload against built-in)', () => {
    expect(resolveComparisonVersion(VERSIONS, 'v2', 'v2', true)).toBe('v2');
  });

  it('defaults to active version for uploads without explicit selection', () => {
    expect(resolveComparisonVersion(VERSIONS, 'v2', null, true)).toBe('v2');
  });

  it('returns null when only one version and no upload', () => {
    const single = [VERSIONS[0]];
    expect(resolveComparisonVersion(single, 'v1', null)).toBeNull();
  });

  it('returns first version when active is unknown', () => {
    expect(resolveComparisonVersion(VERSIONS, 'v99', null)).toBe('v1');
  });
});

// ---------------------------------------------------------------------------
// computeRankDeltas — unit tests with fixture data
// ---------------------------------------------------------------------------

describe('computeRankDeltas', () => {
  it('marks items in current but not comparison as NEW', () => {
    const current = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'CatB', 'z-Score': 1.0 }),
    ];
    const comparison = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 1.5 }),
      // CatB missing
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst']);
    expect(result.get('CatA')!.isNew).toBe(false);
    expect(result.get('CatB')!.isNew).toBe(true);
  });

  it('computes correct rank changes', () => {
    const current = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 3.0 }), // rank 1
      makeRow({ Catalyst: 'CatB', 'z-Score': 1.0 }), // rank 2
    ];
    const comparison = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 1.0 }), // rank 2
      makeRow({ Catalyst: 'CatB', 'z-Score': 2.0 }), // rank 1
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst']);
    // CatA: was rank 2, now rank 1 → moved up by 1
    expect(result.get('CatA')!.rankChange).toBe(1);
    // CatB: was rank 1, now rank 2 → moved down by 1
    expect(result.get('CatB')!.rankChange).toBe(-1);
  });

  it('uses alphabetical tie-breaking for equal medians', () => {
    const current = [
      makeRow({ Catalyst: 'Zebra', 'z-Score': 1.0 }),
      makeRow({ Catalyst: 'Alpha', 'z-Score': 1.0 }),
    ];
    const comparison = [
      makeRow({ Catalyst: 'Zebra', 'z-Score': 1.0 }),
      makeRow({ Catalyst: 'Alpha', 'z-Score': 1.0 }),
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst']);
    // Alpha comes first alphabetically → rank 1
    expect(result.get('Alpha')!.currentRank).toBe(1);
    expect(result.get('Zebra')!.currentRank).toBe(2);
    // Same in comparison → no rank change
    expect(result.get('Alpha')!.rankChange).toBe(0);
    expect(result.get('Zebra')!.rankChange).toBe(0);
  });

  it('handles compound keys with multiple reactant types', () => {
    const current = [
      makeRow({ Catalyst: 'CatA', Ligand: 'LigX', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'CatA', Ligand: 'LigY', 'z-Score': 1.0 }),
    ];
    const comparison = [
      makeRow({ Catalyst: 'CatA', Ligand: 'LigX', 'z-Score': 1.5 }),
      makeRow({ Catalyst: 'CatA', Ligand: 'LigY', 'z-Score': 1.5 }),
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst', 'Ligand']);
    expect(result.has('CatA / LigX')).toBe(true);
    expect(result.has('CatA / LigY')).toBe(true);
    expect(result.get('CatA / LigX')!.isNew).toBe(false);
  });

  it('skips rows with null/NaN z-Score', () => {
    const current = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'CatB', 'z-Score': NaN }),
    ];
    const comparison = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 1.0 }),
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst']);
    expect(result.has('CatA')).toBe(true);
    expect(result.has('CatB')).toBe(false); // no valid z-Score
  });

  it('returns empty map for empty inputs', () => {
    expect(computeRankDeltas([], [], ['Catalyst']).size).toBe(0);
  });

  it('does not include items only in comparison (dropped items)', () => {
    const current = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 2.0 }),
    ];
    const comparison = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 1.5 }),
      makeRow({ Catalyst: 'CatB', 'z-Score': 3.0 }), // only in comparison
      makeRow({ Catalyst: 'CatC', 'z-Score': 0.5 }), // only in comparison
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst']);
    // Only CatA should appear — dropped items are not reported
    expect(result.size).toBe(1);
    expect(result.has('CatA')).toBe(true);
    expect(result.has('CatB')).toBe(false);
    expect(result.has('CatC')).toBe(false);
  });

  it('groups null catalyst values under (no value)', () => {
    const current = [
      makeRow({ Catalyst: null, 'z-Score': 2.0 }),
      makeRow({ Catalyst: null, 'z-Score': 1.0 }),
      makeRow({ Catalyst: 'CatA', 'z-Score': 0.5 }),
    ];
    const comparison = [
      makeRow({ Catalyst: null, 'z-Score': 1.5 }),
      makeRow({ Catalyst: 'CatA', 'z-Score': 0.8 }),
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst']);
    expect(result.has('(no value)')).toBe(true);
    expect(result.get('(no value)')!.isNew).toBe(false);
    // (no value) median: current=[1,2]→1.5, comparison=[1.5]→1.5
    expect(result.get('(no value)')!.currentMedian).toBe(1.5);
    expect(result.get('(no value)')!.comparisonMedian).toBe(1.5);
  });

  it('computes correct median from multiple rows per group', () => {
    const current = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 1.0 }),
      makeRow({ Catalyst: 'CatA', 'z-Score': 3.0 }),
      makeRow({ Catalyst: 'CatA', 'z-Score': 5.0 }),
      makeRow({ Catalyst: 'CatB', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'CatB', 'z-Score': 4.0 }),
    ];
    const comparison = [
      makeRow({ Catalyst: 'CatA', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'CatA', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'CatB', 'z-Score': 10.0 }),
    ];
    const result = computeRankDeltas(current, comparison, ['Catalyst']);
    // CatA current: median([1,3,5]) = 3.0, comparison: median([2,2]) = 2.0
    expect(result.get('CatA')!.currentMedian).toBe(3.0);
    expect(result.get('CatA')!.comparisonMedian).toBe(2.0);
    expect(result.get('CatA')!.medianDelta).toBeCloseTo(1.0);
    // CatB current: median([2,4]) = 3.0, comparison: median([10]) = 10.0
    expect(result.get('CatB')!.currentMedian).toBe(3.0);
    expect(result.get('CatB')!.comparisonMedian).toBe(10.0);
    expect(result.get('CatB')!.medianDelta).toBeCloseTo(-7.0);
  });
});

// ---------------------------------------------------------------------------
// Integration tests with actual parquet data
// ---------------------------------------------------------------------------

describe('comparison with parquet data', () => {
  it('Cu(MeCN)4BF4 is not NEW in either direction', async () => {
    const v1Rows = await loadParquet('v1.parquet');
    const v2Rows = await loadParquet('v2.parquet');

    const params: FilterParams = {
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
      fgA: [], fgB: [],
      copperFilter: 'include', excludeScaleup: true, includeNullCategories: true,
      minEln: 5, topnZscore: 5, maxComponents: 10,
    };

    const v1Filtered = filterData(v1Rows, params);
    const v2Filtered = filterData(v2Rows, params);

    // v1 → v2 direction
    const map1to2 = computeRankDeltas(v1Filtered.rows, v2Filtered.rows, ['Catalyst']);
    expect(map1to2.get('Cu(MeCN)4BF4')!.isNew).toBe(false);
    expect(map1to2.get('Cu(MeCN)4BF4')!.currentRank).toBe(1);

    // v2 → v1 direction
    const map2to1 = computeRankDeltas(v2Filtered.rows, v1Filtered.rows, ['Catalyst']);
    expect(map2to1.get('Cu(MeCN)4BF4')!.isNew).toBe(false);
    expect(map2to1.get('Cu(MeCN)4BF4')!.currentRank).toBe(1);
  });

  it('split by reactantTypes: per-panel filtering keeps top catalysts', async () => {
    const v1Rows = await loadParquet('v1.parquet');
    const v2Rows = await loadParquet('v2.parquet');

    const baseParams: FilterParams = {
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst', 'Ligand'],
      fgA: [], fgB: [],
      copperFilter: 'include', excludeScaleup: true, includeNullCategories: true,
      minEln: 5, topnZscore: 5, maxComponents: 10,
    };

    // Panel uses single reactant type (split mode)
    const panelParams: FilterParams = { ...baseParams, reactantTypes: ['Catalyst'] };
    const v1Panel = filterData(v1Rows, panelParams);
    const v2Panel = filterData(v2Rows, panelParams);

    // Comparison filtered with panel's reactantTypes (the fix)
    const rankMap = computeRankDeltas(v1Panel.rows, v2Panel.rows, ['Catalyst']);
    const cu = rankMap.get('Cu(MeCN)4BF4');
    expect(cu).toBeDefined();
    expect(cu!.isNew).toBe(false);
    expect(cu!.currentRank).toBe(1);

    // Verify the old buggy approach would have failed
    const fullFiltered = filterData(v2Rows, baseParams);
    const buggyMap = computeRankDeltas(v1Panel.rows, fullFiltered.rows, ['Catalyst']);
    const cuBuggy = buggyMap.get('Cu(MeCN)4BF4');
    expect(cuBuggy === undefined || cuBuggy.isNew).toBe(true);
  });

  it('rank ordering is deterministic for tied medians', async () => {
    const v1Rows = await loadParquet('v1.parquet');
    const v2Rows = await loadParquet('v2.parquet');

    const params: FilterParams = {
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
      fgA: [], fgB: [],
      copperFilter: 'include', excludeScaleup: true, includeNullCategories: true,
      minEln: 5, topnZscore: 5, maxComponents: 10,
    };

    const v1Filtered = filterData(v1Rows, params);
    const v2Filtered = filterData(v2Rows, params);

    const map1 = computeRankDeltas(v1Filtered.rows, v2Filtered.rows, ['Catalyst']);
    const map2 = computeRankDeltas(v1Filtered.rows, v2Filtered.rows, ['Catalyst']);

    // Same input → same ranks (deterministic)
    for (const [name, delta] of map1) {
      expect(map2.get(name)!.currentRank).toBe(delta.currentRank);
      expect(map2.get(name)!.comparisonRank).toBe(delta.comparisonRank);
    }

    // Items with equal medians should be alphabetically ordered
    const entries = [...map1.entries()].sort((a, b) => a[1].currentRank - b[1].currentRank);
    for (let i = 1; i < entries.length; i++) {
      const prev = entries[i - 1][1];
      const curr = entries[i][1];
      if (prev.currentMedian === curr.currentMedian) {
        // Equal medians → alphabetical order
        expect(entries[i - 1][0].localeCompare(entries[i][0])).toBeLessThan(0);
      }
    }
  });
});
