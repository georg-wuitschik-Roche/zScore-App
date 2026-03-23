/**
 * Heatmap configuration builder.
 *
 * Creates Plotly data + layout objects for a heatmap of median z-Scores.
 * Port of plot_utils.create_heatmap() from Python.
 *
 * Color scale: blue (low) → white (median) → red (high)
 * Bounds: 5th to 95th percentile of valid median values.
 */

import type { Row } from '../data/types';
import type { Data, Layout } from 'plotly.js';
import type { PlotConfig } from './types';

export type { PlotConfig };

function median(arr: number[]): number {
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 !== 0 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function percentile(arr: number[], p: number): number {
  const s = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

/**
 * Build Plotly heatmap config from filtered rows.
 *
 * First reactant type → y-axis (ordered by median ascending, best on top)
 * Second reactant type → x-axis (ordered by median descending)
 */
export function createHeatmapConfig(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean = false,
): PlotConfig {
  if (rows.length === 0 || reactantTypes.length < 2) {
    return { data: [], layout: {} };
  }

  const yCol = reactantTypes[0]; // first → y-axis
  const xCol = reactantTypes[1]; // second → x-axis

  // Collect z-scores and ELN counts per (x, y) cell
  const cellScores = new Map<string, number[]>();
  const cellElns = new Map<string, Set<string>>();

  for (const row of rows) {
    const x = String(row[xCol] ?? '(no value)');
    const y = String(row[yCol] ?? '(no value)');
    const z = row['z-Score'];
    if (z === null || z === undefined || isNaN(z)) continue;

    const key = `${x}|||${y}`;
    if (!cellScores.has(key)) cellScores.set(key, []);
    cellScores.get(key)!.push(z);

    if (!cellElns.has(key)) cellElns.set(key, new Set());
    if (row.ELN_ID) cellElns.get(key)!.add(row.ELN_ID);
  }

  // Compute median per y-category for ordering
  const yMedians = new Map<string, number>();
  const xMedians = new Map<string, number>();
  const yVals = new Set<string>();
  const xVals = new Set<string>();

  for (const [key] of cellScores) {
    const [x, y] = key.split('|||');
    xVals.add(x);
    yVals.add(y);
  }

  // Compute overall median per y-category
  for (const y of yVals) {
    const allScores: number[] = [];
    for (const x of xVals) {
      const scores = cellScores.get(`${x}|||${y}`);
      if (scores) allScores.push(...scores);
    }
    if (allScores.length > 0) yMedians.set(y, median(allScores));
  }

  // Compute overall median per x-category
  for (const x of xVals) {
    const allScores: number[] = [];
    for (const y of yVals) {
      const scores = cellScores.get(`${x}|||${y}`);
      if (scores) allScores.push(...scores);
    }
    if (allScores.length > 0) xMedians.set(x, median(allScores));
  }

  // Y-axis: ascending median (best performers at top)
  const yArr = Array.from(yVals).sort(
    (a, b) => (yMedians.get(a) ?? 0) - (yMedians.get(b) ?? 0),
  );

  // X-axis: descending median (best performers on left)
  const xArr = Array.from(xVals).sort(
    (a, b) => (xMedians.get(b) ?? 0) - (xMedians.get(a) ?? 0),
  );

  // Build z matrix (median per cell) and ELN count matrix
  const zMatrix: (number | null)[][] = [];
  const textMatrix: string[][] = [];
  const elnMatrix: number[][] = [];

  for (const y of yArr) {
    const zRow: (number | null)[] = [];
    const textRow: string[] = [];
    const elnRow: number[] = [];

    for (const x of xArr) {
      const scores = cellScores.get(`${x}|||${y}`);
      const elns = cellElns.get(`${x}|||${y}`);

      if (scores && scores.length > 0) {
        const med = median(scores);
        zRow.push(med);
        textRow.push(med.toFixed(2));
        elnRow.push(elns?.size ?? 0);
      } else {
        zRow.push(null);
        textRow.push('');
        elnRow.push(0);
      }
    }

    zMatrix.push(zRow);
    textMatrix.push(textRow);
    elnMatrix.push(elnRow);
  }

  // Color scale: 5th percentile (blue) → median (white) → 95th percentile (red)
  const validValues = zMatrix.flat().filter((v): v is number => v !== null);

  let colorscale: [number, string][];
  let zmin: number;
  let zmax: number;

  if (validValues.length > 0) {
    zmin = percentile(validValues, 5);
    zmax = percentile(validValues, 95);
    const zmid = median(validValues);
    const midNorm = zmax > zmin ? (zmid - zmin) / (zmax - zmin) : 0.5;
    colorscale = [
      [0, 'blue'],
      [Math.max(0.01, Math.min(0.99, midNorm)), 'white'],
      [1, 'red'],
    ];
  } else {
    zmin = 0;
    zmax = 1;
    colorscale = [
      [0, 'blue'],
      [0.5, 'white'],
      [1, 'red'],
    ];
  }

  const fontSize = presentationMode ? 18 : 14;
  const height = Math.max(800, yArr.length * 80);

  // Format customdata as [[elnCount]] per cell for hovertemplate
  const customdata = elnMatrix.map((row) => row.map((n) => [n]));

  const monoFont = '"JetBrains Mono", "Fira Code", monospace';

  const data: Data[] = [
    {
      type: 'heatmap' as const,
      z: zMatrix,
      x: xArr,
      y: yArr,
      colorscale,
      zmin,
      zmax,
      xgap: 2,
      ygap: 2,
      showscale: true,
      text: textMatrix as unknown as string[],
      texttemplate: '%{text}',
      textfont: { size: presentationMode ? 14 : 11, color: 'black' },
      colorbar: {
        title: {
          text: 'Median z-Score',
          font: { size: fontSize, family: monoFont },
        },
        tickfont: { family: monoFont },
      },
      hoverongaps: false,
      customdata: customdata as unknown as number[][],
      hovertemplate:
        '<span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px">REAGENTS</span><br>' +
        '<b>%{y}</b><br>' +
        '<b>%{x}</b><br><br>' +
        '<span style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px">RESULTS</span><br>' +
        'Median z-Score: <b>%{z:.3f}</b><br>' +
        'ELNs: <b>%{customdata[0]}</b>' +
        '<extra></extra>',
      hoverlabel: {
        bgcolor: '#fff',
        bordercolor: '#e0e0e0',
        font: { size: 14, family: '"JetBrains Mono", monospace', color: '#222' },
        align: 'left' as const,
      },
    },
  ];

  const layout: Partial<Layout> = {
    title: {
      text: `Median z-Score: ${yCol} vs ${xCol}`,
      font: {
        size: presentationMode ? 28 : 20,
        family: '"JetBrains Mono", "Fira Code", monospace',
        color: '#999',
      },
    },
    xaxis: {
      title: { text: xCol, font: { size: fontSize + 2, family: monoFont }, standoff: 20 },
      tickangle: -45,
      tickfont: { size: presentationMode ? 16 : 13, family: monoFont },
      side: 'bottom',
      automargin: true,
      showgrid: false,
    },
    yaxis: {
      title: { text: yCol, font: { size: fontSize + 2, family: monoFont }, standoff: 20 },
      tickfont: { size: presentationMode ? 16 : 13, family: monoFont },
      automargin: true,
      showgrid: false,
    },
    height,
    margin: { t: 60, b: 180, l: 220, r: 30 },
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
    font: {
      family: '"DM Sans", "Helvetica Neue", sans-serif',
      size: fontSize,
    },
  };

  return { data, layout };
}
