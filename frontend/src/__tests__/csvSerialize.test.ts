/**
 * Tests for CSV serialization.
 *
 * The important property is round-trip: anything toCSV writes must come back
 * out of parseCSVText unchanged, since the app both exports and re-imports CSV.
 */

import { describe, it, expect } from 'vitest';
import { escapeCsvCell, toCSV } from '../data/csvSerialize';
import { parseCSVText } from '../data/loader';

describe('escapeCsvCell', () => {
  it('leaves plain values untouched', () => {
    expect(escapeCsvCell('DMF')).toBe('DMF');
    expect(escapeCsvCell('Pd(OAc)2')).toBe('Pd(OAc)2');
  });

  it('quotes values containing a comma', () => {
    expect(escapeCsvCell('2,6-lutidine')).toBe('"2,6-lutidine"');
  });

  it('quotes and doubles embedded quotes', () => {
    expect(escapeCsvCell('1,1"-bis')).toBe('"1,1""-bis"');
    expect(escapeCsvCell('say "hi"')).toBe('"say ""hi"""');
  });

  it('quotes values containing newlines', () => {
    expect(escapeCsvCell('line1\nline2')).toBe('"line1\nline2"');
    expect(escapeCsvCell('line1\r\nline2')).toBe('"line1\r\nline2"');
  });

  it('renders null and undefined as empty', () => {
    expect(escapeCsvCell(null)).toBe('');
    expect(escapeCsvCell(undefined)).toBe('');
  });

  it('stringifies numbers, including zero', () => {
    expect(escapeCsvCell(1.42)).toBe('1.42');
    expect(escapeCsvCell(0)).toBe('0');
  });
});

describe('toCSV', () => {
  it('escapes headers as well as cells', () => {
    const csv = toCSV(['a,b', 'c'], [{ 'a,b': 1, c: 2 }]);
    expect(csv).toBe('"a,b",c\n1,2');
  });

  it('emits a column per header in order, ignoring extra row keys', () => {
    const csv = toCSV(['b', 'a'], [{ a: 'x', b: 'y', unused: 'z' }]);
    expect(csv).toBe('b,a\ny,x');
  });

  it('emits a header-only line for no rows', () => {
    expect(toCSV(['a', 'b'], [])).toBe('a,b');
  });
});

describe('toCSV → parseCSVText round-trip', () => {
  it('preserves values containing commas, quotes and newlines', async () => {
    const headers = ['ELN_ID', 'Base', 'Catalyst', 'Solvent', 'z-Score'];
    const original = {
      ELN_ID: 'ELN001',
      Base: '2,6-lutidine',
      Catalyst: 'say "hi"',
      Solvent: 'line1\nline2',
      'z-Score': '1.42',
    };

    const parsed = await parseCSVText(toCSV(headers, [original]));

    expect(parsed).toHaveLength(1);
    expect(parsed[0].Base).toBe('2,6-lutidine');
    expect(parsed[0].Catalyst).toBe('say "hi"');
    expect(parsed[0].Solvent).toBe('line1\nline2');
  });

  // Regressions against the old comma-only escaper. PapaParse recovers most of
  // its malformed output, so these two are the cases that genuinely broke:
  // an unescaped newline split one row into two, and a value that was already
  // quoted came back with its quotes stripped.
  it('keeps a value containing a newline in a single row', async () => {
    const parsed = await parseCSVText(toCSV(['Base', 'z-Score'], [{ Base: 'line1\nline2', 'z-Score': '1.0' }]));
    expect(parsed).toHaveLength(1);
    expect(parsed[0].Base).toBe('line1\nline2');
  });

  it('preserves quotes that are part of the value', async () => {
    const parsed = await parseCSVText(toCSV(['Base'], [{ Base: '"Et3N"' }]));
    expect(parsed[0].Base).toBe('"Et3N"');
  });
});
