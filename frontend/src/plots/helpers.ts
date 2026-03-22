/**
 * Shared helpers for distribution plot builders (boxplot, violin).
 *
 * Extracts grouping, sorting, hover text, median overlay, and layout logic
 * so that each plot type only needs to define its trace-specific properties.
 */

import type { Row } from '../data/types';
import type { Data, Layout } from 'plotly.js';
import { createColorMapping } from './colors';
import type { PlotConfig } from './types';

// ── Formatting helpers ────────────────────────────────────────────────

/** Safe string for display — null/undefined → '' */
export function s(val: unknown): string {
  if (val === null || val === undefined || val === '') return '';
  return String(val);
}

/** Format z-Score to 3 decimal places */
export function fmtZ(val: unknown): string {
  if (val === null || val === undefined) return '';
  const n = Number(val);
  return isNaN(n) ? '' : n.toFixed(3);
}

/** Format area to 2 decimal places with % */
export function fmtArea(val: unknown): string {
  if (val === null || val === undefined) return '';
  const n = Number(val);
  return isNaN(n) ? '' : n.toFixed(2) + '%';
}

/** Compute median of a numeric array */
export function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

// ── Prepared data structures ──────────────────────────────────────────

export interface PreparedGroup {
  name: string;
  rows: Row[];
  zScores: number[];
  medianVal: number;
  color: string;
  elnCount: number;
  hoverText: string[];
}

export interface PreparedData {
  groups: PreparedGroup[];
  categoryOrder: string[];
  layout: Partial<Layout>;
}

// ── Core preparation logic ────────────────────────────────────────────

/**
 * Prepare all shared data for a distribution plot.
 *
 * Groups rows by reactant types (compound key when multiple are selected),
 * sorts by descending median, computes colors/hover text/layout.
 * Returns null for empty inputs.
 */
export function prepareDistributionData(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
): PreparedData | null {
  if (rows.length === 0 || reactantTypes.length === 0) return null;

  const groupCol = reactantTypes.length === 1
    ? reactantTypes[0]
    : reactantTypes.join(' / ');

  // Group rows by category value(s)
  const groupMap = new Map<string, Row[]>();
  for (const row of rows) {
    const key = reactantTypes
      .map((col) => String(row[col] ?? '(no value)'))
      .join(' / ');
    const z = row['z-Score'];
    if (z === null || z === undefined || isNaN(z)) continue;
    if (!groupMap.has(key)) groupMap.set(key, []);
    groupMap.get(key)!.push(row);
  }

  // Sort groups by median z-Score descending
  const sorted = Array.from(groupMap.entries())
    .map(([name, groupRows]) => {
      const zScores = groupRows.map((r) => r['z-Score'] as number);
      return { name, rows: groupRows, zScores, medianVal: median(zScores) };
    })
    .sort((a, b) => b.medianVal - a.medianVal);

  // Reverse for Plotly y-axis (bottom-to-top)
  const categoryOrder = sorted.map((g) => g.name).reverse();

  // Color mapping: ELN density → light/dark shade (palette from first reactant type)
  const colorMap = createColorMapping(reactantTypes[0], rows, reactantTypes);

  // Count unique ELNs per category
  const elnCounts = new Map<string, number>();
  for (const { name, rows: groupRows } of sorted) {
    const elns = new Set<string>();
    for (const r of groupRows) {
      if (r.ELN_ID) elns.add(r.ELN_ID);
    }
    elnCounts.set(name, elns.size);
  }

  // Build prepared groups with hover text
  const groups: PreparedGroup[] = sorted.map(({ name, rows: groupRows, zScores, medianVal }) => {
    const color = colorMap.get(name) ?? '#999';
    const elnCount = elnCounts.get(name) ?? 0;

    const hoverText = groupRows.map((row) => {
      const reagentLines = [
        ['Catalyst', row.Catalyst],
        ['Solvent', row.Solvent],
        ['Base', row.Base],
        ['Ligand', row.Ligand],
        ['Additive', row.Additive],
        ['Coupling Reagent', row['Coupling Reagent']],
        ['FG A', row['FG A']],
        ['FG B', row['FG B']],
        ['Secondary Solvent', row['Secondary Solvent']],
      ]
        .filter(([, v]) => v !== null && v !== undefined && v !== '')
        .map(([k, v]) => `<span style="color:#999">${k}:</span> ${s(v)}`)
        .join('<br>');

      return (
        `<span style="color:#999;font-size:12px;letter-spacing:0.05em">EXPERIMENT</span><br>` +
        `<b style="font-size:15px">${s(row.ELN_ID)}</b>` +
        `<span style="color:#999"> · Plate ${s(row.PLATENUMBER)} · ${s(row.Coordinate)}</span><br>` +
        `<br>` +
        `<span style="color:#999;font-size:12px;letter-spacing:0.05em">RESULTS</span><br>` +
        `z-Score: <b>${fmtZ(row['z-Score'])}</b> · Area: ${fmtArea(row.AREA_TOTAL_REDUCED)}<br>` +
        `<br>` +
        `<span style="color:#999;font-size:12px;letter-spacing:0.05em">REAGENTS</span><br>` +
        reagentLines +
        `<br><br>` +
        `<span style="color:#999;font-size:12px;letter-spacing:0.05em">REACTION</span><br>` +
        `${s(row['Reaction Type'])} · <b>${elnCount} ELNs</b>`
      );
    });

    return { name, rows: groupRows, zScores, medianVal, color, elnCount, hoverText };
  });

  // Layout
  const fontSize = presentationMode ? 18 : 14;
  const numCategories = sorted.length;
  const height = Math.max(800, numCategories * 110);

  const layout: Partial<Layout> = {
    title: {
      text: `z-Score Distribution by ${groupCol}`,
      font: {
        size: presentationMode ? 28 : 20,
        family: '"JetBrains Mono", "Fira Code", monospace',
        color: '#999',
      },
    },
    xaxis: {
      title: {
        text: 'z-Score',
        font: {
          size: presentationMode ? 24 : 18,
          family: '"JetBrains Mono", "Fira Code", monospace',
        },
      },
      zeroline: true,
      zerolinecolor: '#ccc',
      gridcolor: '#d0d0d0',
      showgrid: true,
    },
    yaxis: {
      automargin: true,
      categoryorder: 'array',
      categoryarray: categoryOrder,
      showgrid: false,
      tickfont: { size: presentationMode ? 20 : 14, family: '"JetBrains Mono", "Fira Code", monospace' },
    },
    height,
    showlegend: false,
    margin: { t: 60, b: 80, l: 200, r: 50 },
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
    font: {
      family: '"DM Sans", "Helvetica Neue", sans-serif',
      size: fontSize,
    },
  };

  return { groups, categoryOrder, layout };
}

// ── Shared trace builders ─────────────────────────────────────────────

/** Invisible median marker — shows clean tooltip on distribution hover */
export function buildMedianTrace(name: string, medianVal: number, n: number): Data {
  return {
    type: 'scatter' as const,
    x: [medianVal],
    y: [name],
    mode: 'markers' as const,
    marker: { color: 'rgba(0,0,0,0)', size: 20 },
    showlegend: false,
    hovertemplate: `<b>${name}</b><br>Median: ${medianVal.toFixed(3)}<br>n = ${n}<extra></extra>`,
    hoverlabel: {
      bgcolor: '#fff',
      bordercolor: '#e0e0e0',
      font: { size: 14, family: '"JetBrains Mono", monospace', color: '#222' },
      align: 'left' as const,
    },
  };
}

/** Shared hover label style for distribution traces */
export const HOVER_LABEL_STYLE = {
  bgcolor: '#fff',
  bordercolor: '#e0e0e0',
  font: { size: 14, family: '"JetBrains Mono", "Fira Code", monospace', color: '#222' },
  align: 'left' as const,
} as const;

/** Build a PlotConfig from prepared data, using a trace builder for each group */
export function buildDistributionConfig(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
  buildTrace: (group: PreparedGroup) => Data,
): PlotConfig {
  const prepared = prepareDistributionData(rows, reactantTypes, presentationMode);
  if (!prepared) return { data: [], layout: {} };

  const data: Data[] = prepared.groups
    .map((group) => [buildTrace(group), buildMedianTrace(group.name, group.medianVal, group.zScores.length)])
    .flat();

  return { data, layout: prepared.layout };
}
