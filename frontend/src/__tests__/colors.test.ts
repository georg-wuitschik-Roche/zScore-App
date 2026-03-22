/**
 * Tests for color mapping functions from plots/colors.ts.
 *
 * Validates hex interpolation, color mapping creation, and ELN density behavior.
 */

import { describe, it, expect } from 'vitest';
import { interpolateHex, createColorMapping, BASE_COLOURS } from '../plots/colors';
import type { Row } from '../data/types';

// ---------------------------------------------------------------------------
// Helper: create minimal Row for color testing
// ---------------------------------------------------------------------------

function makeRow(overrides: Partial<Row>): Row {
  return {
    ELN_ID: 'ELN001',
    PLATENUMBER: '1',
    Coordinate: 'A1',
    AREA_TOTAL_REDUCED: null,
    Additive: null,
    Base: null,
    Catalyst: null,
    'Coupling Reagent': null,
    Solvent: null,
    Ligand: null,
    'Secondary Solvent': null,
    'Tertiary Solvent': null,
    'Reaction Type': 'Buchwald-Hartwig',
    'FG A': null,
    'FG B': null,
    FG_sorted: null,
    FG_PAIR_SORTED: null,
    'z-Score': 1.0,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// interpolateHex
// ---------------------------------------------------------------------------

describe('interpolateHex', () => {
  it('factor=0 returns first color', () => {
    expect(interpolateHex('#000000', '#ffffff', 0)).toBe('#000000');
  });

  it('factor=1 returns second color', () => {
    expect(interpolateHex('#000000', '#ffffff', 1)).toBe('#ffffff');
  });

  it('factor=0.5 between black and white gives midpoint grey', () => {
    const result = interpolateHex('#000000', '#ffffff', 0.5);
    // Math.round(127.5) = 128 = 0x80
    expect(result).toBe('#808080');
  });

  it('factor=0.5 between red and blue gives purple-ish', () => {
    const result = interpolateHex('#ff0000', '#0000ff', 0.5);
    // R: 255 + (0-255)*0.5 = 128, G: 0, B: 0 + (255-0)*0.5 = 128
    expect(result).toBe('#800080');
  });

  it('result is always a valid 7-char hex string', () => {
    const factors = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1];
    for (const f of factors) {
      const result = interpolateHex('#123456', '#abcdef', f);
      expect(result).toMatch(/^#[0-9a-f]{6}$/);
      expect(result).toHaveLength(7);
    }
  });

  it('handles same color for both inputs', () => {
    const result = interpolateHex('#ff5500', '#ff5500', 0.5);
    expect(result).toBe('#ff5500');
  });

  it('handles factor=0.25 correctly', () => {
    const result = interpolateHex('#000000', '#ffffff', 0.25);
    // 0 + 255 * 0.25 = 63.75 → 64 = 0x40
    expect(result).toBe('#404040');
  });
});

// ---------------------------------------------------------------------------
// createColorMapping
// ---------------------------------------------------------------------------

describe('createColorMapping', () => {
  it('returns a Map', () => {
    const rows: Row[] = [
      makeRow({ Catalyst: 'Pd(OAc)2' }),
    ];
    const result = createColorMapping('Catalyst', rows);
    expect(result).toBeInstanceOf(Map);
  });

  it('uses blue range for Catalyst', () => {
    expect(BASE_COLOURS.Catalyst).toBeDefined();
    expect(BASE_COLOURS.Catalyst.light).toBe('#89CFF1');
    expect(BASE_COLOURS.Catalyst.dark).toBe('#003A6B');
  });

  it('uses grey for unknown/unsupported category', () => {
    const rows: Row[] = [
      makeRow({ 'Unknown Column': 'X' } as Partial<Row>),
    ];
    // 'Unknown Column' is not in BASE_COLOURS → defaults to grey
    const result = createColorMapping('Unknown Column', rows);
    expect(result.size).toBeGreaterThan(0);
    // With single category, factor is 0.5 → midpoint of grey
    const color = result.values().next().value;
    expect(color).toMatch(/^#[0-9a-f]{6}$/);
  });

  it('all same ELN count gives factor 0.5 (midpoint color)', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'ELN001', Catalyst: 'A' }),
      makeRow({ ELN_ID: 'ELN002', Catalyst: 'B' }),
    ];
    // Both have 1 ELN each → minElns === maxElns → factor = 0.5
    const result = createColorMapping('Catalyst', rows);
    const colorA = result.get('A')!;
    const colorB = result.get('B')!;
    const expected = interpolateHex(
      BASE_COLOURS.Catalyst.light,
      BASE_COLOURS.Catalyst.dark,
      0.5,
    );
    expect(colorA).toBe(expected);
    expect(colorB).toBe(expected);
  });

  it('varying ELN counts gives lighter for fewer, darker for more', () => {
    const rows: Row[] = [
      // Category A: 1 ELN
      makeRow({ ELN_ID: 'ELN001', Catalyst: 'A' }),
      // Category B: 3 ELNs
      makeRow({ ELN_ID: 'ELN001', Catalyst: 'B' }),
      makeRow({ ELN_ID: 'ELN002', Catalyst: 'B' }),
      makeRow({ ELN_ID: 'ELN003', Catalyst: 'B' }),
    ];
    const result = createColorMapping('Catalyst', rows);
    const colorA = result.get('A')!;
    const colorB = result.get('B')!;

    // A has factor=0 (min), B has factor=1 (max)
    const expectedA = interpolateHex(
      BASE_COLOURS.Catalyst.light,
      BASE_COLOURS.Catalyst.dark,
      0,
    );
    const expectedB = interpolateHex(
      BASE_COLOURS.Catalyst.light,
      BASE_COLOURS.Catalyst.dark,
      1,
    );
    expect(colorA).toBe(expectedA); // lightest
    expect(colorB).toBe(expectedB); // darkest
  });

  it('null category values become "(no value)" key', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'ELN001', Catalyst: null }),
    ];
    const result = createColorMapping('Catalyst', rows);
    expect(result.has('(no value)')).toBe(true);
  });

  it('creates one entry per unique category value', () => {
    const rows: Row[] = [
      makeRow({ ELN_ID: 'ELN001', Catalyst: 'A' }),
      makeRow({ ELN_ID: 'ELN002', Catalyst: 'A' }),
      makeRow({ ELN_ID: 'ELN003', Catalyst: 'B' }),
      makeRow({ ELN_ID: 'ELN004', Catalyst: 'C' }),
    ];
    const result = createColorMapping('Catalyst', rows);
    expect(result.size).toBe(3);
    expect(result.has('A')).toBe(true);
    expect(result.has('B')).toBe(true);
    expect(result.has('C')).toBe(true);
  });

  it('uses green range for Solvent', () => {
    expect(BASE_COLOURS.Solvent).toBeDefined();
    expect(BASE_COLOURS.Solvent.light).toBe('#90EE90');
    expect(BASE_COLOURS.Solvent.dark).toBe('#006400');

    const rows: Row[] = [
      makeRow({ ELN_ID: 'ELN001', Solvent: 'DMF' }),
    ];
    const result = createColorMapping('Solvent', rows);
    const color = result.get('DMF')!;
    const expected = interpolateHex('#90EE90', '#006400', 0.5);
    expect(color).toBe(expected);
  });

  it('uses orange range for Base', () => {
    expect(BASE_COLOURS.Base.light).toBe('#FFB347');
    expect(BASE_COLOURS.Base.dark).toBe('#CC5500');
  });

  it('handles three categories with graduated ELN counts', () => {
    const rows: Row[] = [
      // Category A: 1 ELN
      makeRow({ ELN_ID: 'E1', Catalyst: 'A' }),
      // Category B: 2 ELNs
      makeRow({ ELN_ID: 'E1', Catalyst: 'B' }),
      makeRow({ ELN_ID: 'E2', Catalyst: 'B' }),
      // Category C: 3 ELNs
      makeRow({ ELN_ID: 'E1', Catalyst: 'C' }),
      makeRow({ ELN_ID: 'E2', Catalyst: 'C' }),
      makeRow({ ELN_ID: 'E3', Catalyst: 'C' }),
    ];
    const result = createColorMapping('Catalyst', rows);
    const colorA = result.get('A')!;
    const colorB = result.get('B')!;
    const colorC = result.get('C')!;

    // A: factor=0, B: factor=0.5, C: factor=1
    expect(colorA).toBe(interpolateHex(BASE_COLOURS.Catalyst.light, BASE_COLOURS.Catalyst.dark, 0));
    expect(colorB).toBe(interpolateHex(BASE_COLOURS.Catalyst.light, BASE_COLOURS.Catalyst.dark, 0.5));
    expect(colorC).toBe(interpolateHex(BASE_COLOURS.Catalyst.light, BASE_COLOURS.Catalyst.dark, 1));
  });
});
