/**
 * Boxplot configuration builder.
 *
 * Creates Plotly data + layout objects from filtered Row data.
 * Port of plot_utils.create_boxplot() from Python.
 */

import type { Row } from '../data/types';
import type { Data, Layout } from 'plotly.js';
import { createColorMapping } from './colors';
import type { PlotConfig } from './types';

export type { PlotConfig };

/** Safe string for display — null/undefined → '' */
function s(val: unknown): string {
  if (val === null || val === undefined || val === '') return '';
  return String(val);
}

/** Format z-Score to 3 decimal places */
function fmtZ(val: unknown): string {
  if (val === null || val === undefined) return '';
  const n = Number(val);
  return isNaN(n) ? '' : n.toFixed(3);
}

/** Format area to 2 decimal places with % */
function fmtArea(val: unknown): string {
  if (val === null || val === undefined) return '';
  const n = Number(val);
  return isNaN(n) ? '' : n.toFixed(2) + '%';
}

/** Compute median of a numeric array */
function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Build Plotly boxplot config from filtered rows.
 *
 * Groups z-Score values by the selected reactant type(s) and creates
 * one box trace per unique category value, sorted by descending median.
 * Each data point has a rich hover tooltip with experiment details.
 */
export function createBoxplotConfig(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean = false,
): PlotConfig {
  if (rows.length === 0 || reactantTypes.length === 0) {
    return { data: [], layout: {} };
  }

  const groupCol = reactantTypes[0];

  // Group rows by category value
  const groups = new Map<string, Row[]>();
  for (const row of rows) {
    const key = String(row[groupCol] ?? '(no value)');
    const z = row['z-Score'];
    if (z === null || z === undefined || isNaN(z)) continue;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(row);
  }

  // Sort groups by median z-Score descending
  const sorted = Array.from(groups.entries())
    .map(([name, groupRows]) => {
      const zScores = groupRows.map((r) => r['z-Score'] as number);
      return { name, rows: groupRows, zScores, median: median(zScores) };
    })
    .sort((a, b) => b.median - a.median);

  // Reverse for Plotly y-axis (bottom-to-top)
  const categoryOrder = sorted.map((g) => g.name).reverse();

  // Color mapping: ELN density → light/dark shade
  const colorMap = createColorMapping(groupCol, rows);

  // Count unique ELNs per category for hover
  const elnCounts = new Map<string, number>();
  for (const { name, rows: groupRows } of sorted) {
    const elns = new Set<string>();
    for (const r of groupRows) {
      if (r.ELN_ID) elns.add(r.ELN_ID);
    }
    elnCounts.set(name, elns.size);
  }

  // Build one trace per category with per-point hover text
  const data: Data[] = sorted.map(({ name, rows: groupRows, zScores }) => {
    const color = colorMap.get(name) ?? '#999';
    const elnCount = elnCounts.get(name) ?? 0;

    // Build styled hover text — reagent lines only shown when populated
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

    // Compute median for the overlay trace
    const sorted = [...zScores].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const medianVal = sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];

    // Box trace — hover on data points only
    const boxTrace = {
      type: 'box' as const,
      x: zScores,
      y: Array(zScores.length).fill(name),
      orientation: 'h' as const,
      name,
      boxpoints: 'all' as const,
      jitter: 0.3,
      pointpos: -1.5,
      boxmean: false,
      marker: { color, size: 6, opacity: 0.5 },
      line: { color: '#333', width: 1.5 },
      fillcolor: color,
      showlegend: false,
      text: hoverText,
      hoverinfo: 'text' as const,
      hoveron: 'points' as const,
      hoverlabel: {
        bgcolor: '#fff',
        bordercolor: '#e0e0e0',
        font: { size: 14, family: '"JetBrains Mono", "Fira Code", monospace', color: '#222' },
        align: 'left' as const,
      },
    };

    // Invisible median marker — shows clean tooltip on box hover
    const medianTrace = {
      type: 'scatter' as const,
      x: [medianVal],
      y: [name],
      mode: 'markers' as const,
      marker: { color: 'rgba(0,0,0,0)', size: 20 },
      showlegend: false,
      hovertemplate: `<b>${name}</b><br>Median: ${medianVal.toFixed(3)}<br>n = ${zScores.length}<extra></extra>`,
      hoverlabel: {
        bgcolor: '#fff',
        bordercolor: '#e0e0e0',
        font: { size: 14, family: '"JetBrains Mono", monospace', color: '#222' },
        align: 'left' as const,
      },
    };

    return [boxTrace, medianTrace];
  }).flat();

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
      title: { text: 'z-Score', font: { size: fontSize } },
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
    margin: { t: 60, b: 60, l: 200, r: 30 },
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
    font: {
      family: '"DM Sans", "Helvetica Neue", sans-serif',
      size: fontSize,
    },
  };

  return { data, layout };
}
