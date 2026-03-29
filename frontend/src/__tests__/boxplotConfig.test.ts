/**
 * Tests for the boxplot configuration builder from plots/boxplot.ts.
 *
 * Validates the PlotConfig output structure, height scaling, category ordering,
 * presentation mode font sizes, empty data handling, and trace properties.
 */

import { describe, it, expect } from 'vitest';
import { createBoxplotConfig } from '../plots/boxplot';
import type { Row } from '../data/types';
import type { BoxPlotData } from 'plotly.js';

// ---------------------------------------------------------------------------
// Helper: create minimal Row for boxplot testing
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

// Small fixture with known z-Scores
const FIXTURE: Row[] = [
  // Catalyst A: z-Scores 1.0, 2.0, 3.0 → median 2.0
  makeRow({ ELN_ID: 'ELN001', Catalyst: 'A', 'z-Score': 1.0 }),
  makeRow({ ELN_ID: 'ELN002', Catalyst: 'A', 'z-Score': 2.0 }),
  makeRow({ ELN_ID: 'ELN003', Catalyst: 'A', 'z-Score': 3.0 }),
  // Catalyst B: z-Scores 4.0, 5.0, 6.0 → median 5.0
  makeRow({ ELN_ID: 'ELN004', Catalyst: 'B', 'z-Score': 4.0 }),
  makeRow({ ELN_ID: 'ELN005', Catalyst: 'B', 'z-Score': 5.0 }),
  makeRow({ ELN_ID: 'ELN006', Catalyst: 'B', 'z-Score': 6.0 }),
  // Catalyst C: z-Scores -1.0, 0.0, 1.0 → median 0.0
  makeRow({ ELN_ID: 'ELN007', Catalyst: 'C', 'z-Score': -1.0 }),
  makeRow({ ELN_ID: 'ELN008', Catalyst: 'C', 'z-Score': 0.0 }),
  makeRow({ ELN_ID: 'ELN009', Catalyst: 'C', 'z-Score': 1.0 }),
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('createBoxplotConfig', () => {
  describe('output structure', () => {
    it('returns object with data and layout properties', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      expect(config).toHaveProperty('data');
      expect(config).toHaveProperty('layout');
    });

    it('data is a non-empty array of Plotly traces', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      expect(Array.isArray(config.data)).toBe(true);
      expect(config.data.length).toBeGreaterThan(0);
    });
  });

  describe('layout height', () => {
    it('height is at least 800 (base height)', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      expect(config.layout.height).toBeGreaterThanOrEqual(800);
    });

    it('height scales with number of categories', () => {
      // 3 categories → height = max(800, 3 * 110) = 800
      const config3 = createBoxplotConfig(FIXTURE, ['Catalyst']);

      // Create more categories to push height beyond 800
      const manyRows: Row[] = [];
      for (let i = 0; i < 10; i++) {
        manyRows.push(
          makeRow({ Catalyst: `Cat${i}`, ELN_ID: `ELN${i}`, 'z-Score': i * 1.0 }),
        );
      }
      const config10 = createBoxplotConfig(manyRows, ['Catalyst']);
      // 10 * 110 = 1100 > 800
      expect(config10.layout.height).toBeGreaterThanOrEqual(1100);
      expect(config10.layout.height!).toBeGreaterThan(config3.layout.height!);
    });
  });

  describe('category ordering', () => {
    it('categories are ordered by median z-Score descending', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      // Medians: B=5.0, A=2.0, C=0.0
      // Sorted descending: B, A, C
      // categoryarray is reversed for Plotly y-axis (bottom-to-top): C, A, B
      const categoryArray = (config.layout as Record<string, unknown> & { yaxis: { categoryarray: string[] } }).yaxis.categoryarray;
      expect(categoryArray).toEqual(['C', 'A', 'B']);
    });
  });

  describe('presentation mode', () => {
    it('with presentationMode=true, font sizes are larger', () => {
      const normal = createBoxplotConfig(FIXTURE, ['Catalyst'], false);
      const presentation = createBoxplotConfig(FIXTURE, ['Catalyst'], true);

      const normalFontSize = (normal.layout.font as { size: number }).size;
      const presFontSize = (presentation.layout.font as { size: number }).size;
      expect(presFontSize).toBeGreaterThan(normalFontSize);
    });

    it('presentationMode=true increases title font size', () => {
      const normal = createBoxplotConfig(FIXTURE, ['Catalyst'], false);
      const presentation = createBoxplotConfig(FIXTURE, ['Catalyst'], true);

      const normalTitleSize = (normal.layout.title as { font: { size: number } }).font.size;
      const presTitleSize = (presentation.layout.title as { font: { size: number } }).font.size;
      expect(presTitleSize).toBeGreaterThan(normalTitleSize);
    });

    it('presentationMode=false uses standard font size (14)', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst'], false);
      expect((config.layout.font as { size: number }).size).toBe(14);
    });

    it('presentationMode=true uses larger font size (18)', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst'], true);
      expect((config.layout.font as { size: number }).size).toBe(18);
    });
  });

  describe('empty data handling', () => {
    it('empty rows returns config with no traces', () => {
      const config = createBoxplotConfig([], ['Catalyst']);
      expect(config.data).toEqual([]);
    });

    it('empty reactantTypes returns config with no traces', () => {
      const config = createBoxplotConfig(FIXTURE, []);
      expect(config.data).toEqual([]);
    });
  });

  describe('trace properties', () => {
    it('box traces have horizontal orientation', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      const boxTraces = config.data.filter(
        (d) => (d as Record<string, unknown>).type === 'box',
      );
      expect(boxTraces.length).toBeGreaterThan(0);
      for (const trace of boxTraces) {
        expect((trace as BoxPlotData).orientation).toBe('h');
      }
    });

    it('box traces have x values (z-Score)', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      const boxTraces = config.data.filter(
        (d) => (d as Record<string, unknown>).type === 'box',
      );
      for (const trace of boxTraces) {
        const x = (trace as BoxPlotData).x as number[];
        expect(Array.isArray(x)).toBe(true);
        expect(x.length).toBeGreaterThan(0);
        expect(x.every((v) => typeof v === 'number')).toBe(true);
      }
    });

    it('hover customdata contains z-Score value and hovertemplate references it', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      const boxTraces = config.data.filter(
        (d) => (d as Record<string, unknown>).type === 'box',
      );
      for (const trace of boxTraces) {
        const rec = trace as Record<string, unknown>;
        const customdata = rec.customdata as string[][];
        expect(Array.isArray(customdata)).toBe(true);
        for (const row of customdata) {
          // Index 3 is the formatted z-Score value
          expect(row[3]).toMatch(/^-?\d+\.\d{3}$/);
        }
        expect(typeof rec.hovertemplate).toBe('string');
        expect(rec.hovertemplate as string).toContain('z-Score');
      }
    });

    it('each category produces a box trace and a median scatter trace', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      // 3 categories → 3 box traces + 3 scatter traces = 6 total
      const boxCount = config.data.filter(
        (d) => (d as Record<string, unknown>).type === 'box',
      ).length;
      const scatterCount = config.data.filter(
        (d) => (d as Record<string, unknown>).type === 'scatter',
      ).length;
      expect(boxCount).toBe(3);
      expect(scatterCount).toBe(3);
    });

    it('box traces have showlegend=false', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      for (const trace of config.data) {
        expect((trace as Record<string, unknown>).showlegend).toBe(false);
      }
    });

    it('title includes the grouping column name', () => {
      const config = createBoxplotConfig(FIXTURE, ['Catalyst']);
      const titleText = (config.layout.title as { text: string }).text;
      expect(titleText).toContain('Catalyst');
    });

    it('rows with null z-Score are excluded from traces', () => {
      const rowsWithNull: Row[] = [
        makeRow({ Catalyst: 'A', 'z-Score': 1.0 }),
        makeRow({ Catalyst: 'A', 'z-Score': null }),
        makeRow({ Catalyst: 'A', 'z-Score': 3.0 }),
      ];
      const config = createBoxplotConfig(rowsWithNull, ['Catalyst']);
      const boxTraces = config.data.filter(
        (d) => (d as Record<string, unknown>).type === 'box',
      );
      // Only 2 valid z-Score values
      const x = (boxTraces[0] as BoxPlotData).x as number[];
      expect(x).toHaveLength(2);
    });

    it('null category values display as "(no value)"', () => {
      const rows: Row[] = [
        makeRow({ Catalyst: null, 'z-Score': 2.0 }),
      ];
      const config = createBoxplotConfig(rows, ['Catalyst']);
      const boxTraces = config.data.filter(
        (d) => (d as Record<string, unknown>).type === 'box',
      );
      const y = (boxTraces[0] as BoxPlotData).y as string[];
      expect(y[0]).toBe('(no value)');
    });
  });
});
