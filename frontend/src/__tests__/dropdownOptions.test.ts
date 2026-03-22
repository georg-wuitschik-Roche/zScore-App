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
import { parseCSVText } from '../data/loader';
import {
  getReactantOptions,
  getFgOptions,
  getFgBOptionsConditioned,
} from '../data/dropdownOptions';
import type { Row } from '../data/types';

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

beforeAll(() => {
  const csvPath = resolve(__dirname, '../../public/data/z-score-peaks.csv');
  dataset = parseCSVText(readFileSync(csvPath, 'utf-8'));
});

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
        const actual = getReactantOptions(dataset, [reactionType]);
        expect(actual.slice().sort()).toEqual(
          expected.reactant_availability.slice().sort(),
        );
      });

      it('FG options match', () => {
        const actual = getFgOptions(dataset, [reactionType]);
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
              dataset,
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
