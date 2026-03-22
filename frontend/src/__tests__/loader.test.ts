/**
 * Tests for the data loader — parseCSVText function.
 *
 * Validates CSV parsing, delimiter auto-detection, null normalization,
 * numeric conversion, and FG_PAIR_SORTED computation.
 */

import { describe, it, expect } from 'vitest';
import { parseCSVText } from '../data/loader';

// ---------------------------------------------------------------------------
// Helper: build a complete CSV row string with all required columns
// ---------------------------------------------------------------------------

const REQUIRED_HEADERS = [
  'ELN_ID', 'PLATENUMBER', 'Coordinate', 'AREA_TOTAL_REDUCED',
  'Base', 'Catalyst', 'Solvent', 'Ligand', 'Additive',
  'Coupling Reagent', 'Secondary Solvent', 'Tertiary Solvent',
  'Reaction Type', 'FG A', 'FG B', 'FG_sorted', 'z-Score', 'output_column',
];

const HEADER_LINE = REQUIRED_HEADERS.join(',');

/** Quote a CSV field if it contains a comma. */
function csvField(val: string): string {
  return val.includes(',') ? `"${val}"` : val;
}

function makeRow(overrides: Record<string, string> = {}): string {
  const defaults: Record<string, string> = {
    ELN_ID: 'ELN001',
    PLATENUMBER: '1',
    Coordinate: 'A1',
    AREA_TOTAL_REDUCED: '50.0',
    Base: 'K3PO4',
    Catalyst: 'Pd(OAc)2',
    Solvent: 'DMF',
    Ligand: 'XPhos',
    Additive: '',
    'Coupling Reagent': '',
    'Secondary Solvent': '',
    'Tertiary Solvent': '',
    'Reaction Type': 'Buchwald-Hartwig',
    'FG A': 'ArBr',
    'FG B': 'RNH2',
    FG_sorted: 'ArBr, RNH2',
    'z-Score': '1.23',
    output_column: 'Catalyst',
  };
  const merged = { ...defaults, ...overrides };
  return REQUIRED_HEADERS.map((h) => csvField(merged[h] ?? '')).join(',');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('parseCSVText', () => {
  describe('comma-delimited CSV', () => {
    it('parses correct number of rows', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ ELN_ID: 'ELN001', 'z-Score': '1.23' }),
        makeRow({ ELN_ID: 'ELN002', 'z-Score': '2.34' }),
        makeRow({ ELN_ID: 'ELN003', 'z-Score': '3.45' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows).toHaveLength(3);
    });

    it('preserves column values accurately', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ ELN_ID: 'ELN042', Catalyst: 'CuI', Base: 'Cs2CO3' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0].ELN_ID).toBe('ELN042');
      expect(rows[0].Catalyst).toBe('CuI');
      expect(rows[0].Base).toBe('Cs2CO3');
    });
  });

  describe('delimiter auto-detection', () => {
    it('auto-detects semicolon delimiter', () => {
      const csv = [
        REQUIRED_HEADERS.join(';'),
        REQUIRED_HEADERS.map((h) => {
          if (h === 'ELN_ID') return 'ELN001';
          if (h === 'z-Score') return '1.5';
          if (h === 'Catalyst') return 'Pd(OAc)2';
          if (h === 'Reaction Type') return 'Buchwald-Hartwig';
          if (h === 'FG A') return 'ArBr';
          if (h === 'FG B') return 'RNH2';
          if (h === 'FG_sorted') return 'ArBr, RNH2';
          if (h === 'PLATENUMBER') return '1';
          if (h === 'Coordinate') return 'A1';
          return '';
        }).join(';'),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows).toHaveLength(1);
      expect(rows[0].ELN_ID).toBe('ELN001');
      expect(rows[0].Catalyst).toBe('Pd(OAc)2');
    });

    it('auto-detects tab delimiter', () => {
      const csv = [
        REQUIRED_HEADERS.join('\t'),
        REQUIRED_HEADERS.map((h) => {
          if (h === 'ELN_ID') return 'ELN001';
          if (h === 'z-Score') return '2.5';
          if (h === 'Catalyst') return 'CuI';
          if (h === 'Reaction Type') return 'Suzuki-Miyaura';
          if (h === 'FG A') return 'ArCl';
          if (h === 'FG B') return 'ArNH2';
          if (h === 'FG_sorted') return 'ArCl, ArNH2';
          if (h === 'PLATENUMBER') return '2';
          if (h === 'Coordinate') return 'B3';
          return '';
        }).join('\t'),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows).toHaveLength(1);
      expect(rows[0].ELN_ID).toBe('ELN001');
      expect(rows[0].Catalyst).toBe('CuI');
    });
  });

  describe('null normalization', () => {
    it('normalizes empty string to null for categorical columns', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ Additive: '', Base: '', Catalyst: '', Ligand: '' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0].Additive).toBeNull();
      expect(rows[0].Base).toBeNull();
      expect(rows[0].Catalyst).toBeNull();
      expect(rows[0].Ligand).toBeNull();
    });

    it('normalizes "nan" string to null', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ Additive: 'nan', Base: 'NaN', 'Coupling Reagent': 'nan' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0].Additive).toBeNull();
      expect(rows[0].Base).toBeNull();
      expect(rows[0]['Coupling Reagent']).toBeNull();
    });

    it('normalizes "NaN" string to null', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ Solvent: 'NaN', 'Secondary Solvent': 'NaN' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0].Solvent).toBeNull();
      expect(rows[0]['Secondary Solvent']).toBeNull();
    });

    it('preserves non-null categorical values', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ Catalyst: 'Pd(OAc)2', Base: 'K3PO4', Solvent: 'DMF' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0].Catalyst).toBe('Pd(OAc)2');
      expect(rows[0].Base).toBe('K3PO4');
      expect(rows[0].Solvent).toBe('DMF');
    });
  });

  describe('z-Score conversion', () => {
    it('converts z-Score strings to numbers', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ 'z-Score': '1.23' }),
        makeRow({ 'z-Score': '-0.5' }),
        makeRow({ 'z-Score': '0' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0]['z-Score']).toBe(1.23);
      expect(rows[1]['z-Score']).toBe(-0.5);
      expect(rows[2]['z-Score']).toBe(0);
    });

    it('handles comma-as-decimal separator in z-Score', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ 'z-Score': '1,23' }),
        makeRow({ 'z-Score': '-0,5' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0]['z-Score']).toBe(1.23);
      expect(rows[1]['z-Score']).toBe(-0.5);
    });

    it('handles empty z-Score as null', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ 'z-Score': '' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0]['z-Score']).toBeNull();
    });
  });

  describe('FG_PAIR_SORTED computation', () => {
    it('uses FG_sorted as FG_PAIR_SORTED when present', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ 'FG A': 'ArBr', 'FG B': 'RNH2', FG_sorted: 'ArBr, RNH2' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(rows[0].FG_PAIR_SORTED).toBe('ArBr, RNH2');
    });

    it('computes FG_PAIR_SORTED from FG A + FG B when FG_sorted is missing', () => {
      // Build CSV without FG_sorted column
      const headers = REQUIRED_HEADERS.filter((h) => h !== 'FG_sorted');
      const headerLine = headers.join(',');
      const values = headers.map((h) => {
        if (h === 'ELN_ID') return 'ELN001';
        if (h === 'FG A') return 'RNH2';
        if (h === 'FG B') return 'ArBr';
        if (h === 'z-Score') return '1.0';
        if (h === 'PLATENUMBER') return '1';
        if (h === 'Coordinate') return 'A1';
        if (h === 'Reaction Type') return 'Buchwald-Hartwig';
        return '';
      }).join(',');

      const csv = [headerLine, values].join('\n');
      const rows = parseCSVText(csv);
      // Sorted alphabetically: ArBr < RNH2
      expect(rows[0].FG_PAIR_SORTED).toBe('ArBr, RNH2');
    });

    it('computes FG_PAIR_SORTED with correct sorting', () => {
      const csv = [
        HEADER_LINE,
        makeRow({ 'FG A': 'RNH2', 'FG B': 'ArBr', FG_sorted: '' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      // Empty FG_sorted falls through to computation: sorted → ArBr, RNH2
      expect(rows[0].FG_PAIR_SORTED).toBe('ArBr, RNH2');
    });
  });

  describe('BigInt handling', () => {
    it('converts BigInt-like values to numbers', () => {
      // The cleanRow function converts BigInt to Number.
      // In CSV parsing, values arrive as strings, so we test that
      // numeric columns are properly parsed as numbers.
      const csv = [
        HEADER_LINE,
        makeRow({ AREA_TOTAL_REDUCED: '12345' }),
      ].join('\n');

      const rows = parseCSVText(csv);
      expect(typeof rows[0].AREA_TOTAL_REDUCED).toBe('number');
      expect(rows[0].AREA_TOTAL_REDUCED).toBe(12345);
    });
  });

  describe('edge cases', () => {
    it('returns empty array for empty CSV', () => {
      const rows = parseCSVText('');
      expect(rows).toHaveLength(0);
    });

    it('returns empty array for header-only CSV', () => {
      const rows = parseCSVText(HEADER_LINE);
      expect(rows).toHaveLength(0);
    });

    it('handles rows with missing columns gracefully', () => {
      const csv = `ELN_ID,z-Score,Catalyst,FG A,FG B
ELN001,1.23,Pd(OAc)2,ArBr,RNH2
ELN002,,CuI,ArCl,ArNH2`;

      const rows = parseCSVText(csv);
      expect(rows).toHaveLength(2);
      expect(rows[0].ELN_ID).toBe('ELN001');
      expect(rows[0]['z-Score']).toBe(1.23);
      expect(rows[1]['z-Score']).toBeNull();
      expect(rows[1].Catalyst).toBe('CuI');
    });
  });
});
