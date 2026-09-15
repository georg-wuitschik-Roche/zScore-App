/**
 * Golden fixture tests for dropdown option computation.
 *
 * Validates that the TypeScript dropdown functions produce the same
 * reactant availability, FG options, and conditioned FG B options
 * as the Python reference implementation.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { describe, it, expect, beforeAll } from 'vitest';
import { parseDataset } from '../data/loader';
import {
  getReactantOptions,
  getFgOptions,
  getFgBOptionsConditioned,
  getReactionTypeElnCounts,
  getReactantElnCounts,
  getFgAElnCounts,
  getFgBElnCounts,
} from '../data/dropdownOptions';
import {
  filterByReactionTypes,
  filterByReactantColumns,
  filterCopper,
  filterPrecomplexed,
  filterFgA,
  filterFgB,
} from '../data/filterSteps';
import { countElns } from '../data/filterChain';
import { DEFAULTS } from '../data/types';
import type { FilterParams, Row } from '../data/types';

// ---------------------------------------------------------------------------
// Golden fixture types
// ---------------------------------------------------------------------------

interface DropdownGoldenEntry {
  row_count: number;
  reactant_availability: string[];
  fg_all_options: string[];
  fg_b_conditioned: Record<string, string[]>;
}

type DropdownGolden = Record<string, DropdownGoldenEntry>;

// ---------------------------------------------------------------------------
// Load dataset and golden fixtures
// ---------------------------------------------------------------------------

let dataset: Row[];

const goldenDir = resolve(__dirname, '../../golden');
const golden: DropdownGolden = JSON.parse(
  readFileSync(resolve(goldenDir, 'dropdown_conditioning.json'), 'utf-8'),
);

beforeAll(async () => {
  const parquetPath = resolve(__dirname, '../../public/data/z-score-peaks.parquet');
  const buffer = readFileSync(parquetPath);
  dataset = await parseDataset(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
});

// Pre-filter dataset once per reaction type to avoid redundant 67K-row scans
const preFiltered = new Map<string, Row[]>();
function rowsFor(rt: string): Row[] {
  let rows = preFiltered.get(rt);
  if (!rows) {
    rows = dataset.filter((r) => r['Reaction Type'] === rt);
    preFiltered.set(rt, rows);
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Dropdown conditioning (golden fixtures)', () => {
  const reactionTypes = Object.keys(golden);

  it(`has ${reactionTypes.length} reaction types in golden file`, () => {
    expect(reactionTypes.length).toBeGreaterThan(0);
  });

  for (const reactionType of reactionTypes) {
    const expected = golden[reactionType];

    describe(`${reactionType}`, () => {
      it('reactant availability matches', () => {
        const actual = getReactantOptions(rowsFor(reactionType), [reactionType]);
        expect(actual.slice().sort()).toEqual(
          expected.reactant_availability.slice().sort(),
        );
      });

      it('FG options match', () => {
        const actual = getFgOptions(rowsFor(reactionType), [reactionType]);
        expect(actual.slice().sort()).toEqual(
          expected.fg_all_options.slice().sort(),
        );
      });

      // Conditioned FG B tests: each key is a single FG A value or
      // a "+" separated list of FG A values
      const fgBKeys = Object.keys(expected.fg_b_conditioned);

      if (fgBKeys.length > 0) {
        for (const fgAKey of fgBKeys) {
          const expectedFgB = expected.fg_b_conditioned[fgAKey];
          const fgASelection = fgAKey.split('+');

          it(`FG B conditioned on [${fgASelection.join(', ')}] matches`, () => {
            const actual = getFgBOptionsConditioned(
              rowsFor(reactionType),
              [reactionType],
              fgASelection,
            );
            expect(actual.slice().sort()).toEqual(expectedFgB.slice().sort());
          });
        }
      }
    });
  }
});

describe('Dropdown ELN counts', () => {
  /**
   * Chain steps 1-5 — the filters the counts promise to account for. Selecting
   * an option should land the user exactly here, so this is the ground truth
   * every count is measured against.
   *
   * Deliberately composed here rather than calling `baseFilter` from
   * filterChain: the counting code already calls that, so sharing it would let
   * a missing or misordered step corrupt oracle and subject alike and every
   * assertion would still pass. Keep this independent.
   */
  function baseFilterOracle(rows: Row[], params: FilterParams): Row[] {
    let scoped = filterByReactionTypes(rows, params.reactionTypes);
    scoped = filterByReactantColumns(
      scoped,
      params.reactantTypes,
      params.includeNullCategories,
    );
    scoped = filterCopper(scoped, params.copperFilter);
    scoped = filterPrecomplexed(scoped, params.precomplexedFilter);
    const [afterFgA, fgAList] = filterFgA(scoped, params.fgA);
    const [afterFgB] = filterFgB(afterFgA, params.fgB, fgAList);
    return afterFgB;
  }

  const RT = 'Buchwald-Hartwig';
  const base: FilterParams = { ...DEFAULTS, reactionTypes: [RT] };

  const scenarios: { name: string; params: FilterParams }[] = [
    { name: 'reaction type only', params: base },
    { name: 'with FG A', params: { ...base, fgA: ['ArBr'] } },
    {
      name: 'with FG A and FG B',
      params: { ...base, fgA: ['ArBr'], fgB: ['ArNH2', 'R2NH'] },
    },
    {
      name: 'with a reactant type',
      params: { ...base, fgA: ['ArBr'], reactantTypes: ['Catalyst'] },
    },
    {
      name: 'with copper included and null categories excluded',
      params: {
        ...base,
        copperFilter: 'include',
        includeNullCategories: false,
        reactantTypes: ['Ligand'],
      },
    },
    { name: 'no reaction type selected', params: DEFAULTS },
  ];

  // Reactant counts are excluded: they measure column availability, which only
  // lines up with the reactant filter when null categories are excluded. See
  // the dedicated test below.
  const dimensions = [
    { label: 'reaction type', counts: getReactionTypeElnCounts, key: 'reactionTypes' },
    { label: 'FG A', counts: getFgAElnCounts, key: 'fgA' },
    { label: 'FG B', counts: getFgBElnCounts, key: 'fgB' },
  ] as const;

  for (const { name, params } of scenarios) {
    describe(name, () => {
      // Each count must equal what you actually get by picking that option on
      // top of the current selection — every other active filter included.
      for (const { label, counts, key } of dimensions) {
        it(`${label} counts match picking that option`, () => {
          for (const [option, count] of Object.entries(counts(dataset, params))) {
            const picked = baseFilterOracle(dataset, { ...params, [key]: [option] });
            expect(count, `${label}: ${option}`).toBe(countElns(picked));
          }
        });
      }
    });
  }

  // Reactant counts measure column availability, so they only line up with the
  // reactant filter when null categories are excluded.
  it('reactant counts match picking that column when nulls are excluded', () => {
    const params: FilterParams = {
      ...base,
      includeNullCategories: false,
      fgA: ['ArBr'],
    };
    const counts = getReactantElnCounts(dataset, params);
    expect(Object.keys(counts).length).toBeGreaterThan(0);

    for (const [cat, count] of Object.entries(counts)) {
      const picked = baseFilterOracle(dataset, { ...params, reactantTypes: [cat] });
      expect(count, cat).toBe(countElns(picked));
    }
  });

  // The counts must never exceed what the filters already in effect allow —
  // a reactant column claiming more ELNs than the FG B badge shows is the bug
  // that made the first cut of this feature misleading.
  it('no count exceeds the ELNs surviving the other active filters', () => {
    const params: FilterParams = {
      ...base,
      fgA: ['ArBr'],
      fgB: ['ArNH2', 'R2NH'],
    };
    const ceiling = countElns(baseFilterOracle(dataset, params));
    expect(ceiling).toBeGreaterThan(0);

    for (const [cat, count] of Object.entries(
      getReactantElnCounts(dataset, params),
    )) {
      expect(count, `${cat} -> ${count} > ${ceiling}`).toBeLessThanOrEqual(
        ceiling,
      );
    }
  });

  it('counts only offer options that survive the other filters', () => {
    const params: FilterParams = { ...base, fgA: ['ArBr'] };
    const counts = getFgBElnCounts(dataset, params);
    const options = getFgBOptionsConditioned(rowsFor(RT), [RT], ['ArBr']);
    // Copper exclusion can drop an option entirely, but never add one
    expect(Object.keys(counts).length).toBeGreaterThan(0);
    for (const fg of Object.keys(counts)) {
      expect(options, fg).toContain(fg);
    }
  });

  it('returns an empty map for an empty dataset', () => {
    expect(getReactionTypeElnCounts([], DEFAULTS)).toEqual({});
    expect(getFgAElnCounts([], DEFAULTS)).toEqual({});
    expect(getFgBElnCounts([], base)).toEqual({});
    expect(getReactantElnCounts([], DEFAULTS)).toEqual({});
  });
});
