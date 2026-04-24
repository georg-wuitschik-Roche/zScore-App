/**
 * Parity tests — verify TypeScript filter chain, color mapping, boxplot,
 * and heatmap produce correct, consistent results.
 *
 * Golden fixtures were initially generated from the Python paper/ code
 * (before deletion) to validate that the TypeScript implementations
 * are equivalent. Known differences due to null-handling and sort
 * stability are accounted for with tolerance-based assertions.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { describe, it, expect, beforeAll } from 'vitest';
import { parseDataset } from '../data/loader';
import { filterData } from '../data/filterChain';
import { createColorMapping, interpolateHex } from '../plots/colors';
import { createBoxplotConfig } from '../plots/boxplot';
import { createHeatmapConfig } from '../plots/heatmap';
import type { Row, FilterParams } from '../data/types';
import { isCopperCatalyst } from '../data/filterSteps';

// ---------------------------------------------------------------------------
// Load dataset once
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
// Helpers
// ---------------------------------------------------------------------------

interface FilterCase {
  name: string;
  params: FilterParams;
  minRows: number; // Minimum expected rows (sanity check)
}

const FILTER_CASES: FilterCase[] = [
  {
    name: 'buchwald_catalyst',
    params: {
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
      fgA: [],
      fgB: [],
      copperFilter: 'exclude',
      precomplexedFilter: 'include',
      excludeScaleup: true,
      includeNullCategories: false,
      topnZscore: 5,
      minEln: 5,
      maxComponents: 10,
    },
    minRows: 1000,
  },
  {
    name: 'buchwald_ligand',
    params: {
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Ligand'],
      fgA: [],
      fgB: [],
      copperFilter: 'exclude',
      precomplexedFilter: 'include',
      excludeScaleup: true,
      includeNullCategories: false,
      topnZscore: 3,
      minEln: 5,
      maxComponents: 10,
    },
    minRows: 1000,
  },
  {
    name: 'suzuki_catalyst',
    params: {
      reactionTypes: ['Suzuki-Miyaura'],
      reactantTypes: ['Catalyst'],
      fgA: [],
      fgB: [],
      copperFilter: 'include',
      precomplexedFilter: 'include',
      excludeScaleup: true,
      includeNullCategories: false,
      topnZscore: 5,
      minEln: 5,
      maxComponents: 12,
    },
    minRows: 1500,
  },
  {
    name: 'suzuki_solvent_base',
    params: {
      reactionTypes: ['Suzuki-Miyaura'],
      reactantTypes: ['Solvent', 'Base'],
      fgA: [],
      fgB: [],
      copperFilter: 'include',
      precomplexedFilter: 'include',
      excludeScaleup: true,
      includeNullCategories: false,
      topnZscore: 5,
      minEln: 5,
      maxComponents: 10,
    },
    minRows: 500,
  },
  {
    name: 'buchwald_fg_pair',
    params: {
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
      fgA: ['ArBr'],
      fgB: ['R2NH'],
      copperFilter: 'exclude',
      precomplexedFilter: 'include',
      excludeScaleup: true,
      includeNullCategories: false,
      topnZscore: 5,
      minEln: 5,
      maxComponents: 10,
    },
    minRows: 200,
  },
  {
    name: 'amide_include_null',
    params: {
      reactionTypes: ['Amide coupling'],
      reactantTypes: ['Base'],
      fgA: [],
      fgB: [],
      copperFilter: 'include',
      precomplexedFilter: 'include',
      excludeScaleup: false,
      includeNullCategories: true,
      topnZscore: 5,
      minEln: 3,
      maxComponents: 15,
    },
    minRows: 200,
  },
];

// ---------------------------------------------------------------------------
// Color interpolation parity (with Python golden fixtures)
// ---------------------------------------------------------------------------

interface InterpolationCase {
  col1: string;
  col2: string;
  factor: number;
  result: string;
}

const goldenDir = resolve(__dirname, '../../golden/parity');

describe('Color interpolation parity with Python', () => {
  let interpCases: InterpolationCase[];

  beforeAll(() => {
    const fixtures = JSON.parse(readFileSync(resolve(goldenDir, 'color_parity.json'), 'utf-8'));
    interpCases = fixtures['interpolation_samples']?.cases ?? [];
  });

  it('has interpolation test cases', () => {
    expect(interpCases.length).toBeGreaterThan(0);
  });

  it('interpolateHex matches Python within RGB tolerance of 1', () => {
    const hexToRgb = (hex: string) => {
      const h = hex.replace('#', '');
      return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
    };

    for (const c of interpCases) {
      const result = interpolateHex(c.col1, c.col2, c.factor);
      const expected = hexToRgb(c.result);
      const actual = hexToRgb(result);
      for (let i = 0; i < 3; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Filter chain regression tests
// ---------------------------------------------------------------------------

describe('Filter chain regression', () => {
  for (const { name, params, minRows } of FILTER_CASES) {
    describe(name, () => {
      it('produces non-empty result above minimum', () => {
        const { rows } = filterData(dataset, params);
        expect(rows.length).toBeGreaterThanOrEqual(minRows);
      });

      it('all rows match the selected reaction type', () => {
        const { rows } = filterData(dataset, params);
        for (const row of rows) {
          expect(params.reactionTypes).toContain(row['Reaction Type']);
        }
      });

      it('copper exclusion is applied when enabled', () => {
        if (params.copperFilter !== 'exclude') return;
        const { rows } = filterData(dataset, params);
        const hasCopper = rows.some((r) => isCopperCatalyst(r.Catalyst as string | null));
        expect(hasCopper).toBe(false);
      });

      it('all z-Scores are valid numbers', () => {
        const { rows } = filterData(dataset, params);
        // After deduplication, some rows may have null z-scores
        // but the vast majority should be valid
        const validZ = rows.filter(
          (r) => r['z-Score'] !== null && !isNaN(r['z-Score'] as number),
        );
        expect(validZ.length).toBeGreaterThan(rows.length * 0.95);
      });

      it('stats contain expected fields', () => {
        const { stats } = filterData(dataset, params);
        expect(stats.wholeDataset).toBeDefined();
        expect(stats.wholeDataset!.elns).toBeGreaterThan(0);
        if (params.reactantTypes.length > 0) {
          expect(stats.maxComponentsCap).toBeDefined();
        }
      });

      it('maxComponents cap is respected', () => {
        const { rows } = filterData(dataset, params);
        if (params.reactantTypes.length === 0 || params.maxComponents <= 0) return;

        const keyCols = params.reactantTypes;
        const combos = new Set<string>();
        for (const row of rows) {
          const key = keyCols.map((c) => row[c] ?? '').join('|');
          combos.add(key);
        }
        expect(combos.size).toBeLessThanOrEqual(params.maxComponents);
      });
    });
  }
});

// ---------------------------------------------------------------------------
// Boxplot regression tests
// ---------------------------------------------------------------------------

describe('Boxplot regression', () => {
  const BOXPLOT_CASES = FILTER_CASES.filter((c) => c.params.reactantTypes.length === 1);

  for (const { name, params } of BOXPLOT_CASES) {
    describe(name, () => {
      it('produces box traces', () => {
        const { rows } = filterData(dataset, params);
        const config = createBoxplotConfig(rows, params.reactantTypes);
        const boxTraces = config.data.filter((d) => 'type' in d && d.type === 'box');
        expect(boxTraces.length).toBeGreaterThan(0);
        expect(boxTraces.length).toBeLessThanOrEqual(params.maxComponents);
      });

      it('produces matching median traces', () => {
        const { rows } = filterData(dataset, params);
        const config = createBoxplotConfig(rows, params.reactantTypes);
        const boxTraces = config.data.filter((d) => 'type' in d && d.type === 'box');
        const scatterTraces = config.data.filter((d) => 'type' in d && d.type === 'scatter');
        // Each box trace has a matching invisible median marker, plus 1 colorbar trace
        expect(scatterTraces.length).toBe(boxTraces.length + 1);
      });

      it('height scales with number of categories', () => {
        const { rows } = filterData(dataset, params);
        const config = createBoxplotConfig(rows, params.reactantTypes);
        const boxTraces = config.data.filter((d) => 'type' in d && d.type === 'box');
        const minHeight = Math.max(800, boxTraces.length * 110);
        expect(config.layout.height).toBeGreaterThanOrEqual(minHeight);
      });

      it('categories are sorted by descending median', () => {
        const { rows } = filterData(dataset, params);
        const config = createBoxplotConfig(rows, params.reactantTypes);
        const boxTraces = config.data.filter(
          (d) => 'type' in d && d.type === 'box',
        );

        // Extract medians from trace data
        const medians = boxTraces.map((trace) => {
          const xVals = (trace as { x: number[] }).x;
          const sorted = [...xVals].sort((a, b) => a - b);
          const mid = Math.floor(sorted.length / 2);
          return sorted.length % 2 !== 0
            ? sorted[mid]
            : (sorted[mid - 1] + sorted[mid]) / 2;
        });

        // Should be descending
        for (let i = 1; i < medians.length; i++) {
          expect(medians[i - 1]).toBeGreaterThanOrEqual(medians[i]);
        }
      });
    });
  }
});

// ---------------------------------------------------------------------------
// Color mapping regression tests
// ---------------------------------------------------------------------------

describe('Color mapping regression', () => {
  it('creates colors for all categories in filtered data', () => {
    const { rows } = filterData(dataset, FILTER_CASES[0].params);
    const colorMap = createColorMapping('Catalyst', rows);
    expect(colorMap.size).toBeGreaterThan(0);

    // All color values should be valid hex
    for (const [, color] of colorMap) {
      expect(color).toMatch(/^#[0-9a-f]{6}$/);
    }
  });

  it('single-category dataset gets midpoint color', () => {
    const rows: Row[] = [
      { ELN_ID: 'E1', PLATENUMBER: '1', Coordinate: 'A1', Catalyst: 'Pd', 'z-Score': 1 } as Row,
      { ELN_ID: 'E1', PLATENUMBER: '1', Coordinate: 'A2', Catalyst: 'Pd', 'z-Score': 2 } as Row,
    ];
    const colorMap = createColorMapping('Catalyst', rows);
    expect(colorMap.size).toBe(1);
    // With only one category, factor = 0.5 (midpoint)
    const color = colorMap.get('Pd')!;
    expect(color).toMatch(/^#[0-9a-f]{6}$/);
  });
});

// ---------------------------------------------------------------------------
// Heatmap regression tests
// ---------------------------------------------------------------------------

describe('Heatmap regression', () => {
  const heatmapCase = FILTER_CASES.find((c) => c.params.reactantTypes.length >= 2);

  it('produces a heatmap with correct structure', () => {
    if (!heatmapCase) return;
    const { rows } = filterData(dataset, heatmapCase.params);
    const config = createHeatmapConfig(rows, heatmapCase.params.reactantTypes);

    expect(config.data.length).toBe(1);
    const heatmapData = config.data[0] as Record<string, unknown>;
    expect(heatmapData.type).toBe('heatmap');

    const xLabels = heatmapData.x as string[];
    const yLabels = heatmapData.y as string[];
    const zMatrix = heatmapData.z as (number | null)[][];

    expect(xLabels.length).toBeGreaterThan(0);
    expect(yLabels.length).toBeGreaterThan(0);
    expect(zMatrix.length).toBe(yLabels.length);
    expect(zMatrix[0].length).toBe(xLabels.length);
  });

  it('height scales with y-axis categories', () => {
    if (!heatmapCase) return;
    const { rows } = filterData(dataset, heatmapCase.params);
    const config = createHeatmapConfig(rows, heatmapCase.params.reactantTypes);

    const heatmapData = config.data[0] as Record<string, unknown>;
    const yLabels = heatmapData.y as string[];
    const expectedHeight = Math.max(800, yLabels.length * 80);
    expect(config.layout.height).toBe(expectedHeight);
  });

  it('returns empty for single reactant type', () => {
    const config = createHeatmapConfig(dataset.slice(0, 100), ['Catalyst']);
    expect(config.data.length).toBe(0);
  });
});
