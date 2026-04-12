/**
 * Tests for individual filter step functions from filterSteps.ts.
 *
 * Uses a small fixture dataset (~20 rows) with known values
 * to exercise each of the 10 filter steps plus helpers.
 */

import { describe, it, expect } from 'vitest';
import type { Row } from '../data/types';
import {
  normalizeFgInput,
  median,
  filterByReactionTypes,
  filterByReactantColumns,
  filterCopper,
  isCopperCatalyst,
  filterFgA,
  filterFgB,
  filterScaleupPlates,
  deduplicateBestZscore,
  filterTopNZscore,
  filterMinEln,
  filterMaxComponents,
} from '../data/filterSteps';

// ---------------------------------------------------------------------------
// Fixture dataset (~20 rows)
// ---------------------------------------------------------------------------

function makeRow(overrides: Partial<Row>): Row {
  return {
    ELN_ID: 'ELN001',
    PLATENUMBER: '1',
    Coordinate: 'A1',
    AREA_TOTAL_REDUCED: 50.0,
    Additive: null,
    Base: 'K3PO4',
    Catalyst: 'Pd(OAc)2',
    'Coupling Reagent': null,
    Solvent: 'DMF',
    Ligand: 'XPhos',
    'Secondary Solvent': null,
    'Tertiary Solvent': null,
    'Reaction Type': 'Buchwald-Hartwig',
    'FG A': 'ArBr',
    'FG B': 'RNH2',
    FG_sorted: 'ArBr, RNH2',
    FG_PAIR_SORTED: 'ArBr, RNH2',
    'z-Score': 1.0,
    output_column: 'Catalyst',
    ...overrides,
  };
}

const FIXTURE: Row[] = [
  // Buchwald-Hartwig rows (ELN001-ELN004)
  makeRow({ ELN_ID: 'ELN001', PLATENUMBER: '1', Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'DMF', Ligand: 'XPhos', 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 2.5 }),
  makeRow({ ELN_ID: 'ELN001', PLATENUMBER: '1', Catalyst: 'CuI', Base: 'K3PO4', Solvent: 'DMF', Ligand: 'XPhos', 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 1.5 }),
  makeRow({ ELN_ID: 'ELN002', PLATENUMBER: '2', Catalyst: 'Pd(OAc)2', Base: 'Cs2CO3', Solvent: 'DMF', Ligand: 'SPhos', 'FG A': 'ArCl', 'FG B': 'ArNH2', FG_PAIR_SORTED: 'ArCl, ArNH2', 'z-Score': 3.0 }),
  makeRow({ ELN_ID: 'ELN002', PLATENUMBER: '2', Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'THF', Ligand: 'SPhos', 'FG A': 'ArCl', 'FG B': 'ArNH2', FG_PAIR_SORTED: 'ArCl, ArNH2', 'z-Score': 2.0 }),
  makeRow({ ELN_ID: 'ELN003', PLATENUMBER: '3', Catalyst: 'Pd2(dba)3', Base: 'K3PO4', Solvent: 'DMF', Ligand: null, 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 4.0 }),
  makeRow({ ELN_ID: 'ELN003', PLATENUMBER: '3', Catalyst: null, Base: 'K3PO4', Solvent: 'DMF', Ligand: null, 'FG A': 'ArBr', 'FG B': 'ArNH2', FG_PAIR_SORTED: 'ArBr, ArNH2', 'z-Score': 0.5 }),
  makeRow({ ELN_ID: 'ELN004', PLATENUMBER: '4', Catalyst: 'CuI', Base: 'Cs2CO3', Solvent: 'DMSO', Ligand: null, 'FG A': 'RNH2', 'FG B': 'ArBr', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': -1.0 }),
  // Suzuki-Miyaura rows (ELN005-ELN008)
  makeRow({ ELN_ID: 'ELN005', PLATENUMBER: '5', 'Reaction Type': 'Suzuki-Miyaura', Catalyst: 'Pd(PPh3)4', Base: 'K2CO3', Solvent: 'Dioxane', Ligand: null, 'FG A': 'ArBr', 'FG B': 'ArB(OH)2', FG_PAIR_SORTED: 'ArB(OH)2, ArBr', 'z-Score': 5.0 }),
  makeRow({ ELN_ID: 'ELN005', PLATENUMBER: '5', 'Reaction Type': 'Suzuki-Miyaura', Catalyst: 'CuI', Base: 'K2CO3', Solvent: 'Dioxane', Ligand: null, 'FG A': 'ArBr', 'FG B': 'ArB(OH)2', FG_PAIR_SORTED: 'ArB(OH)2, ArBr', 'z-Score': 0.1 }),
  makeRow({ ELN_ID: 'ELN006', PLATENUMBER: '6', 'Reaction Type': 'Suzuki-Miyaura', Catalyst: 'Pd(PPh3)4', Base: 'Na2CO3', Solvent: 'THF', Ligand: 'PPh3', 'FG A': 'ArCl', 'FG B': 'ArB(OH)2', FG_PAIR_SORTED: 'ArB(OH)2, ArCl', 'z-Score': 3.5 }),
  makeRow({ ELN_ID: 'ELN007', PLATENUMBER: '7', 'Reaction Type': 'Suzuki-Miyaura', Catalyst: 'Pd(OAc)2', Base: 'K2CO3', Solvent: 'DMF', Ligand: 'XPhos', 'FG A': 'ArBr', 'FG B': 'ArB(OH)2', FG_PAIR_SORTED: 'ArB(OH)2, ArBr', 'z-Score': 2.0 }),
  makeRow({ ELN_ID: 'ELN008', PLATENUMBER: '8', 'Reaction Type': 'Suzuki-Miyaura', Catalyst: 'Pd(PPh3)4', Base: 'K2CO3', Solvent: 'Dioxane', Ligand: null, 'FG A': 'ArBr', 'FG B': 'ArB(OH)2', FG_PAIR_SORTED: 'ArB(OH)2, ArBr', 'z-Score': 1.0 }),
  // Scale-up plate: all same reagents (plate 9 under ELN009)
  makeRow({ ELN_ID: 'ELN009', PLATENUMBER: '9', Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'DMF', Ligand: 'XPhos', 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 6.0 }),
  makeRow({ ELN_ID: 'ELN009', PLATENUMBER: '9', Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'DMF', Ligand: 'XPhos', 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 5.5 }),
  // Duplicate reagent combos with different z-Scores (for dedup testing)
  makeRow({ ELN_ID: 'ELN010', PLATENUMBER: '10', Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'DMF', Ligand: 'XPhos', 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 1.0 }),
  makeRow({ ELN_ID: 'ELN010', PLATENUMBER: '10', Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'DMF', Ligand: 'XPhos', 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 3.0 }),
  // Row with null z-Score
  makeRow({ ELN_ID: 'ELN011', PLATENUMBER: '11', Catalyst: 'Pd2(dba)3', Base: 'K3PO4', Solvent: 'DMF', Ligand: null, 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': null }),
  // Row with Additive populated
  makeRow({ ELN_ID: 'ELN001', PLATENUMBER: '1', Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'DMF', Ligand: 'XPhos', Additive: 'LiCl', 'FG A': 'ArBr', 'FG B': 'RNH2', FG_PAIR_SORTED: 'ArBr, RNH2', 'z-Score': 1.8 }),
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

describe('normalizeFgInput', () => {
  it('null returns empty array', () => {
    expect(normalizeFgInput(null)).toEqual([]);
  });

  it('undefined returns empty array', () => {
    expect(normalizeFgInput(undefined)).toEqual([]);
  });

  it('empty string returns empty array', () => {
    expect(normalizeFgInput('')).toEqual([]);
  });

  it('"All" returns empty array', () => {
    expect(normalizeFgInput('All')).toEqual([]);
  });

  it('single FG string returns single-element array', () => {
    expect(normalizeFgInput('ArBr')).toEqual(['ArBr']);
  });

  it('array with "All" filters it out', () => {
    expect(normalizeFgInput(['ArBr', 'All', 'RNH2'])).toEqual(['ArBr', 'RNH2']);
  });

  it('array without "All" returns as-is', () => {
    expect(normalizeFgInput(['ArBr', 'RNH2'])).toEqual(['ArBr', 'RNH2']);
  });

  it('array with only "All" returns empty array', () => {
    expect(normalizeFgInput(['All'])).toEqual([]);
  });
});

describe('median', () => {
  it('empty array returns NaN', () => {
    expect(median([])).toBeNaN();
  });

  it('single element returns that element', () => {
    expect(median([5])).toBe(5);
  });

  it('two elements returns their average', () => {
    expect(median([1, 3])).toBe(2);
  });

  it('three elements returns middle value', () => {
    expect(median([1, 2, 3])).toBe(2);
  });

  it('four elements returns average of middle two', () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });

  it('unsorted input still gives correct median', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
  });

  it('handles negative numbers', () => {
    expect(median([-3, -1, 0])).toBe(-1);
  });
});

// ---------------------------------------------------------------------------
// Step 1: filterByReactionTypes
// ---------------------------------------------------------------------------

describe('filterByReactionTypes', () => {
  it('null/empty returns all rows', () => {
    expect(filterByReactionTypes(FIXTURE, [])).toHaveLength(FIXTURE.length);
  });

  it('single type filters correctly', () => {
    const result = filterByReactionTypes(FIXTURE, ['Buchwald-Hartwig']);
    expect(result.length).toBeGreaterThan(0);
    expect(result.every((r) => r['Reaction Type'] === 'Buchwald-Hartwig')).toBe(true);
  });

  it('multiple types returns union', () => {
    const result = filterByReactionTypes(FIXTURE, ['Buchwald-Hartwig', 'Suzuki-Miyaura']);
    expect(result).toHaveLength(FIXTURE.length);
  });

  it('nonexistent type returns empty result', () => {
    const result = filterByReactionTypes(FIXTURE, ['Nonexistent-Reaction']);
    expect(result).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Step 2: filterByReactantColumns
// ---------------------------------------------------------------------------

describe('filterByReactantColumns', () => {
  it('null reactantTypes returns all', () => {
    expect(filterByReactantColumns(FIXTURE, [], false)).toHaveLength(FIXTURE.length);
  });

  it('includeNull=true returns all rows', () => {
    const result = filterByReactantColumns(FIXTURE, ['Ligand'], true);
    expect(result).toHaveLength(FIXTURE.length);
  });

  it('filters out rows with null in specified columns', () => {
    const result = filterByReactantColumns(FIXTURE, ['Ligand'], false);
    expect(result.length).toBeLessThan(FIXTURE.length);
    expect(result.every((r) => r.Ligand !== null)).toBe(true);
  });

  it('multiple columns: requires all non-null', () => {
    const result = filterByReactantColumns(FIXTURE, ['Ligand', 'Additive'], false);
    expect(result.every((r) => r.Ligand !== null && r.Additive !== null)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Step 3: filterCopper
// ---------------------------------------------------------------------------

describe('filterCopper', () => {
  it('include mode returns all rows', () => {
    expect(filterCopper(FIXTURE, 'include')).toHaveLength(FIXTURE.length);
  });

  it('exclude mode removes rows with copper catalysts', () => {
    const result = filterCopper(FIXTURE, 'exclude');
    const copperRows = FIXTURE.filter((r) => isCopperCatalyst(r.Catalyst as string | null));
    expect(copperRows.length).toBeGreaterThan(0);
    expect(result.every((r) => !isCopperCatalyst(r.Catalyst as string | null))).toBe(true);
    expect(result).toHaveLength(FIXTURE.length - copperRows.length);
  });

  it('exclude mode preserves rows where Catalyst is null', () => {
    const result = filterCopper(FIXTURE, 'exclude');
    const nullCatRows = FIXTURE.filter((r) => r.Catalyst === null);
    expect(nullCatRows.length).toBeGreaterThan(0);
    expect(result.filter((r) => r.Catalyst === null)).toHaveLength(nullCatRows.length);
  });

  it('only mode keeps only rows with copper catalysts', () => {
    const result = filterCopper(FIXTURE, 'only');
    expect(result.length).toBeGreaterThan(0);
    expect(result.every((r) => isCopperCatalyst(r.Catalyst as string | null))).toBe(true);
  });

  it('only mode excludes rows with null Catalyst', () => {
    const result = filterCopper(FIXTURE, 'only');
    expect(result.every((r) => r.Catalyst !== null)).toBe(true);
  });

  it('matches various copper catalyst forms', () => {
    const testRows = [
      makeRow({ Catalyst: 'CuI' }),
      makeRow({ Catalyst: 'CuBr' }),
      makeRow({ Catalyst: 'Cu(OAc)2' }),
      makeRow({ Catalyst: 'Cu(MeCN)4BF4' }),
      makeRow({ Catalyst: 'Copper(I) thiophene-2-carboxylate' }),
      makeRow({ Catalyst: 'Pd(OAc)2, CuI' }),
      makeRow({ Catalyst: 'Pd(OAc)2' }),  // not copper
    ];
    const result = filterCopper(testRows, 'only');
    expect(result).toHaveLength(6);
  });
});

// ---------------------------------------------------------------------------
// Step 4: filterFgA
// ---------------------------------------------------------------------------

describe('filterFgA', () => {
  it('empty list returns all rows and empty fgAList', () => {
    const [result, fgAList] = filterFgA(FIXTURE, []);
    expect(result).toHaveLength(FIXTURE.length);
    expect(fgAList).toEqual([]);
  });

  it('null returns all rows', () => {
    const [result, fgAList] = filterFgA(FIXTURE, null);
    expect(result).toHaveLength(FIXTURE.length);
    expect(fgAList).toEqual([]);
  });

  it('single FG filters rows where FG A or FG B matches', () => {
    const [result, fgAList] = filterFgA(FIXTURE, ['ArCl']);
    expect(fgAList).toEqual(['ArCl']);
    expect(result.length).toBeGreaterThan(0);
    expect(result.every((r) => r['FG A'] === 'ArCl' || r['FG B'] === 'ArCl')).toBe(true);
  });

  it('multiple FGs returns union of matches', () => {
    const [result] = filterFgA(FIXTURE, ['ArCl', 'ArB(OH)2']);
    expect(result.length).toBeGreaterThan(0);
    expect(
      result.every(
        (r) =>
          r['FG A'] === 'ArCl' || r['FG B'] === 'ArCl' ||
          r['FG A'] === 'ArB(OH)2' || r['FG B'] === 'ArB(OH)2',
      ),
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Step 5: filterFgB
// ---------------------------------------------------------------------------

describe('filterFgB', () => {
  it('empty list returns all rows', () => {
    const [result, fgBList] = filterFgB(FIXTURE, [], []);
    expect(result).toHaveLength(FIXTURE.length);
    expect(fgBList).toEqual([]);
  });

  it('with fgAList matches FG_PAIR_SORTED pairs', () => {
    const fgAList = ['ArBr'];
    const [result] = filterFgB(FIXTURE, ['RNH2'], fgAList);
    expect(result.length).toBeGreaterThan(0);
    // All rows should match the pair ArBr,RNH2 (sorted)
    expect(result.every((r) => r.FG_PAIR_SORTED === 'ArBr, RNH2')).toBe(true);
  });

  it('without fgAList matches FG A or FG B directly', () => {
    const [result] = filterFgB(FIXTURE, ['ArB(OH)2'], []);
    expect(result.length).toBeGreaterThan(0);
    expect(
      result.every((r) => r['FG A'] === 'ArB(OH)2' || r['FG B'] === 'ArB(OH)2'),
    ).toBe(true);
  });

  it('with fgAList and multiple fgB values creates all pair combinations', () => {
    const fgAList = ['ArBr'];
    const [result] = filterFgB(FIXTURE, ['RNH2', 'ArNH2'], fgAList);
    expect(result.length).toBeGreaterThan(0);
    const validPairs = new Set(['ArBr, RNH2', 'ArBr, ArNH2']);
    expect(result.every((r) => validPairs.has(r.FG_PAIR_SORTED ?? ''))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Step 6: filterScaleupPlates
// ---------------------------------------------------------------------------

describe('filterScaleupPlates', () => {
  it('false returns all rows', () => {
    expect(filterScaleupPlates(FIXTURE, false)).toHaveLength(FIXTURE.length);
  });

  it('removes plates where no reagent column has >1 unique value', () => {
    // Plate 9 (ELN009) has identical reagents across both rows → scale-up
    const result = filterScaleupPlates(FIXTURE, true);
    const plate9Rows = result.filter(
      (r) => r.ELN_ID === 'ELN009' && r.PLATENUMBER === '9',
    );
    expect(plate9Rows).toHaveLength(0);
  });

  it('keeps plates with reagent variability', () => {
    // Plate 1 (ELN001) has Pd(OAc)2 and CuI → 2 unique Catalysts
    const result = filterScaleupPlates(FIXTURE, true);
    const plate1Rows = result.filter(
      (r) => r.ELN_ID === 'ELN001' && r.PLATENUMBER === '1',
    );
    expect(plate1Rows.length).toBeGreaterThan(0);
  });

  it('keeps plates with variability in any reagent column', () => {
    // Plate 2 (ELN002) has DMF and THF → 2 unique Solvents
    const result = filterScaleupPlates(FIXTURE, true);
    const plate2Rows = result.filter(
      (r) => r.ELN_ID === 'ELN002' && r.PLATENUMBER === '2',
    );
    expect(plate2Rows.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Step 7: deduplicateBestZscore
// ---------------------------------------------------------------------------

describe('deduplicateBestZscore', () => {
  it('keeps row with highest z-Score per reagent combination', () => {
    // ELN010 has two rows with identical reagents: z=1.0 and z=3.0
    const eln010Rows = FIXTURE.filter((r) => r.ELN_ID === 'ELN010');
    expect(eln010Rows).toHaveLength(2);

    const result = deduplicateBestZscore(eln010Rows);
    expect(result).toHaveLength(1);
    expect(result[0]['z-Score']).toBe(3.0);
  });

  it('handles null in groupby columns', () => {
    // Include rows with null Ligand — dedup still works
    const rowsWithNull = FIXTURE.filter(
      (r) => r.Ligand === null && r['z-Score'] !== null,
    );
    expect(rowsWithNull.length).toBeGreaterThan(0);

    const result = deduplicateBestZscore(rowsWithNull);
    // Should produce results without errors
    expect(result.length).toBeGreaterThanOrEqual(1);
    expect(result.length).toBeLessThanOrEqual(rowsWithNull.length);
  });

  it('drops groups where all z-Scores are NaN/null', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'X', Catalyst: 'A', 'z-Score': null }),
    ];
    const result = deduplicateBestZscore(rows);
    expect(result).toHaveLength(0);
  });

  it('preserves rows from different reagent combinations', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'E1', Catalyst: 'A', Solvent: 'S1', 'z-Score': 1.0 }),
      makeRow({ ELN_ID: 'E1', Catalyst: 'B', Solvent: 'S1', 'z-Score': 2.0 }),
    ];
    const result = deduplicateBestZscore(rows);
    expect(result).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Step 8: filterTopNZscore
// ---------------------------------------------------------------------------

describe('filterTopNZscore', () => {
  it('null topN returns all rows', () => {
    expect(filterTopNZscore(FIXTURE, 0, ['Catalyst'], true)).toHaveLength(FIXTURE.length);
  });

  it('topN=1 keeps best per group', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'E1', Catalyst: 'A', 'z-Score': 3.0 }),
      makeRow({ ELN_ID: 'E1', Catalyst: 'A', 'z-Score': 1.0 }),
      makeRow({ ELN_ID: 'E1', Catalyst: 'A', 'z-Score': 2.0 }),
    ];
    const result = filterTopNZscore(rows, 1, ['Catalyst'], true);
    expect(result).toHaveLength(1);
    expect(result[0]['z-Score']).toBe(3.0);
  });

  it('topN=2 keeps top 2 per group', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'E1', Catalyst: 'A', 'z-Score': 3.0 }),
      makeRow({ ELN_ID: 'E1', Catalyst: 'A', 'z-Score': 1.0 }),
      makeRow({ ELN_ID: 'E1', Catalyst: 'A', 'z-Score': 2.0 }),
    ];
    const result = filterTopNZscore(rows, 2, ['Catalyst'], true);
    expect(result).toHaveLength(2);
    const zScores = result.map((r) => r['z-Score']).sort((a, b) => (b ?? 0) - (a ?? 0));
    expect(zScores).toEqual([3.0, 2.0]);
  });

  it('empty reactantTypes returns all', () => {
    expect(filterTopNZscore(FIXTURE, 1, [], true)).toHaveLength(FIXTURE.length);
  });
});

// ---------------------------------------------------------------------------
// Step 9: filterMinEln
// ---------------------------------------------------------------------------

describe('filterMinEln', () => {
  it('null/0 minEln returns all rows', () => {
    expect(filterMinEln(FIXTURE, 0, ['Catalyst'], true)).toHaveLength(FIXTURE.length);
  });

  it('minEln=2 removes groups with <2 unique ELNs', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'E1', 'Reaction Type': 'BH', Catalyst: 'A', 'z-Score': 1.0 }),
      makeRow({ ELN_ID: 'E2', 'Reaction Type': 'BH', Catalyst: 'A', 'z-Score': 2.0 }),
      makeRow({ ELN_ID: 'E1', 'Reaction Type': 'BH', Catalyst: 'B', 'z-Score': 3.0 }),
    ];
    // Group (BH, A) has 2 ELNs (E1, E2) → keep
    // Group (BH, B) has 1 ELN (E1) → remove
    const result = filterMinEln(rows, 2, ['Catalyst'], false);
    expect(result).toHaveLength(2);
    expect(result.every((r) => r.Catalyst === 'A')).toBe(true);
  });

  it('empty reactantTypes returns all rows', () => {
    expect(filterMinEln(FIXTURE, 5, [], true)).toHaveLength(FIXTURE.length);
  });

  it('minEln=1 keeps all groups with at least one ELN', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'E1', 'Reaction Type': 'BH', Catalyst: 'A', 'z-Score': 1.0 }),
      makeRow({ ELN_ID: 'E1', 'Reaction Type': 'BH', Catalyst: 'B', 'z-Score': 2.0 }),
    ];
    const result = filterMinEln(rows, 1, ['Catalyst'], false);
    expect(result).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Step 10: filterMaxComponents
// ---------------------------------------------------------------------------

describe('filterMaxComponents', () => {
  it('null/0 maxComponents returns all rows', () => {
    expect(filterMaxComponents(FIXTURE, 0, ['Catalyst'], true)).toHaveLength(FIXTURE.length);
  });

  it('limits number of unique category groups by median z-Score', () => {
    const rows: Row[] = [
      // Group A: median = 5.0
      makeRow({ Catalyst: 'A', 'z-Score': 5.0 }),
      makeRow({ Catalyst: 'A', 'z-Score': 5.0 }),
      // Group B: median = 3.0
      makeRow({ Catalyst: 'B', 'z-Score': 3.0 }),
      makeRow({ Catalyst: 'B', 'z-Score': 3.0 }),
      // Group C: median = 1.0
      makeRow({ Catalyst: 'C', 'z-Score': 1.0 }),
      makeRow({ Catalyst: 'C', 'z-Score': 1.0 }),
    ];

    const result = filterMaxComponents(rows, 2, ['Catalyst'], true);
    // Should keep top 2 by median: A (5.0) and B (3.0)
    const cats = new Set(result.map((r) => r.Catalyst));
    expect(cats.size).toBe(2);
    expect(cats.has('A')).toBe(true);
    expect(cats.has('B')).toBe(true);
    expect(cats.has('C')).toBe(false);
  });

  it('returns all when maxComponents >= number of groups', () => {
    const rows: Row[] = [
      makeRow({ Catalyst: 'A', 'z-Score': 1.0 }),
      makeRow({ Catalyst: 'B', 'z-Score': 2.0 }),
    ];
    const result = filterMaxComponents(rows, 10, ['Catalyst'], true);
    expect(result).toHaveLength(2);
  });

  it('empty reactantTypes returns all rows', () => {
    expect(filterMaxComponents(FIXTURE, 2, [], true)).toHaveLength(FIXTURE.length);
  });

  it('breaks ties alphabetically', () => {
    const rows: Row[] = [
      makeRow({ Catalyst: 'B', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'A', 'z-Score': 2.0 }),
      makeRow({ Catalyst: 'C', 'z-Score': 2.0 }),
    ];
    // All medians equal (2.0), so alphabetical: A, B, C → keep first 2: A, B
    const result = filterMaxComponents(rows, 2, ['Catalyst'], true);
    const cats = new Set(result.map((r) => r.Catalyst));
    expect(cats.size).toBe(2);
    expect(cats.has('A')).toBe(true);
    expect(cats.has('B')).toBe(true);
  });
});
