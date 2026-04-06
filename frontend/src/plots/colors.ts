/**
 * Color mapping for boxplots — port of plot_utils.py BASE_COLOURS + interpolation.
 *
 * Each reactant type has a light→dark color ramp. Categories with more ELNs
 * get darker shades, giving a visual cue about data density.
 */

import type { Row } from '../data/types';

export const BASE_COLOURS: Record<string, { light: string; dark: string }> = {
  Catalyst: { light: '#89CFF1', dark: '#003A6B' }, // blue
  Solvent: { light: '#90EE90', dark: '#006400' }, // green
  Base: { light: '#FFB347', dark: '#CC5500' }, // orange
  Ligand: { light: '#E6E6FA', dark: '#4B0082' }, // purple
  Additive: { light: '#FFB6C1', dark: '#8B0000' }, // red
  'Coupling Reagent': { light: '#E6E6FA', dark: '#191970' }, // purple-blue
  'Functional Group A': { light: '#FFC0CB', dark: '#C71585' }, // pink
  'Functional Group B': { light: '#87CEEB', dark: '#006994' }, // sky blue
  'Secondary Solvent': { light: '#98FB98', dark: '#228B22' }, // light green
};

export const DEFAULT_COLOURS = { light: '#D3D3D3', dark: '#696969' }; // grey

/** Distinct palette used when multiple reactant types are combined */
export const COMBINED_COLOURS = { light: '#B0BEC5', dark: '#263238' }; // blue-grey → dark slate

/**
 * Linear interpolation between two hex colours (0 ≤ factor ≤ 1).
 */
export function interpolateHex(col1: string, col2: string, factor: number): string {
  const hexToRgb = (hex: string): [number, number, number] => {
    const h = hex.replace('#', '');
    return [
      parseInt(h.slice(0, 2), 16),
      parseInt(h.slice(2, 4), 16),
      parseInt(h.slice(4, 6), 16),
    ];
  };

  const [r1, g1, b1] = hexToRgb(col1);
  const [r2, g2, b2] = hexToRgb(col2);

  const r = Math.round(r1 + (r2 - r1) * factor);
  const g = Math.round(g1 + (g2 - g1) * factor);
  const b = Math.round(b1 + (b2 - b1) * factor);

  return (
    '#' +
    r.toString(16).padStart(2, '0') +
    g.toString(16).padStart(2, '0') +
    b.toString(16).padStart(2, '0')
  );
}

/**
 * Create a color mapping from category value → hex color.
 *
 * Categories with more unique ELNs get darker shades. If all categories
 * have the same ELN count, factor defaults to 0.5 (midpoint).
 *
 * @param category  Reactant type name for palette selection (e.g. "Catalyst")
 * @param rows      Dataset rows
 * @param groupCols Columns to form the compound grouping key (defaults to [category])
 */
export function createColorMapping(
  category: string,
  rows: Row[],
  groupCols?: string[],
  combined?: boolean,
): Map<string, string> {
  const base = combined ? COMBINED_COLOURS : (BASE_COLOURS[category] ?? DEFAULT_COLOURS);
  const cols = groupCols ?? [category];

  // Count unique ELNs per category value
  const elnSets = new Map<string, Set<string>>();
  for (const row of rows) {
    const catVal = cols.map((c) => String(row[c] ?? '(no value)')).join(' / ');
    if (!elnSets.has(catVal)) elnSets.set(catVal, new Set());
    if (row.ELN_ID) elnSets.get(catVal)!.add(row.ELN_ID);
  }

  const counts = new Map<string, number>();
  let maxElns = 0;
  let minElns = Infinity;
  for (const [catVal, elns] of elnSets) {
    const n = elns.size;
    counts.set(catVal, n);
    if (n > maxElns) maxElns = n;
    if (n < minElns) minElns = n;
  }

  const colorMap = new Map<string, string>();
  for (const [catVal, cnt] of counts) {
    const factor =
      maxElns === minElns ? 0.5 : (cnt - minElns) / (maxElns - minElns);
    colorMap.set(catVal, interpolateHex(base.light, base.dark, factor));
  }

  return colorMap;
}

/**
 * Create a color mapping from pre-computed ELN counts (avoids re-iterating rows).
 */
export function createColorMappingFromElnCounts(
  category: string,
  elnCounts: Map<string, number>,
  combined?: boolean,
): Map<string, string> {
  const base = combined ? COMBINED_COLOURS : (BASE_COLOURS[category] ?? DEFAULT_COLOURS);

  let maxElns = 0;
  let minElns = Infinity;
  for (const n of elnCounts.values()) {
    if (n > maxElns) maxElns = n;
    if (n < minElns) minElns = n;
  }

  const colorMap = new Map<string, string>();
  for (const [catVal, cnt] of elnCounts) {
    const factor =
      maxElns === minElns ? 0.5 : (cnt - minElns) / (maxElns - minElns);
    colorMap.set(catVal, interpolateHex(base.light, base.dark, factor));
  }

  return colorMap;
}
