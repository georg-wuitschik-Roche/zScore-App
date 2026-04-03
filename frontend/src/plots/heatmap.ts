/**
 * Heatmap configuration builder.
 *
 * Creates Plotly data + layout objects for a heatmap of median z-Scores.
 * Port of plot_utils.create_heatmap() from Python.
 *
 * Color scale: blue (low) → white (median) → red (high)
 * Bounds: 5th to 95th percentile of valid median values.
 */

import type { Row, RankDelta, ComparisonInfo } from '../data/types';
import type { Data, Layout } from 'plotly.js';
import type { PlotConfig } from './types';
import { buildRankAnnotation, getHoverLabelStyle, median, RANK_BADGE_TICK_PAD } from './helpers';

export type { PlotConfig };

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
/**
 * @param axisRankMaps - Per-axis rank deltas: [0] = y-axis, [1] = x-axis
 */
export function createHeatmapConfig(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean = false,
  axisRankMaps?: Map<string, RankDelta>[] | null,
  isDark = false,
  comparisonInfo?: ComparisonInfo | null,
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
  const ySorted = Array.from(yVals).sort(
    (a, b) => (yMedians.get(a) ?? 0) - (yMedians.get(b) ?? 0),
  );

  // X-axis: descending median (best performers on left)
  const xSorted = Array.from(xVals).sort(
    (a, b) => (xMedians.get(b) ?? 0) - (xMedians.get(a) ?? 0),
  );

  // Build colored rank badge annotations for axis labels
  const rankAnnotations: Partial<Layout>['annotations'] = [];
  if (axisRankMaps && axisRankMaps.length >= 2) {
    const yRankMap = axisRankMaps[0];
    const xRankMap = axisRankMaps[1];
    const badgeSize = presentationMode ? 18 : 13;
    for (const label of ySorted) {
      const delta = yRankMap.get(label);
      if (delta) {
        rankAnnotations.push(buildRankAnnotation(delta, label, 'y', badgeSize, comparisonInfo, isDark));
      }
    }
    for (const label of xSorted) {
      const delta = xRankMap.get(label);
      if (delta) {
        rankAnnotations.push(buildRankAnnotation(delta, label, 'x-top', badgeSize, comparisonInfo, isDark));
      }
    }
  }
  const yArr = ySorted;
  const xArr = xSorted;

  // Build z matrix (median per cell) and ELN count matrix
  // Use original sorted keys for cell lookups, annotated labels for display
  const zMatrix: (number | null)[][] = [];
  const textMatrix: string[][] = [];
  const elnMatrix: number[][] = [];

  for (const y of ySorted) {
    const zRow: (number | null)[] = [];
    const textRow: string[] = [];
    const elnRow: number[] = [];

    for (const x of xSorted) {
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
        `<span style="font-size:11px;color:${isDark ? '#aaa' : '#888'};text-transform:uppercase;letter-spacing:1px">REAGENTS</span><br>` +
        '<b>%{y}</b><br>' +
        '<b>%{x}</b><br><br>' +
        `<span style="font-size:11px;color:${isDark ? '#aaa' : '#888'};text-transform:uppercase;letter-spacing:1px">RESULTS</span><br>` +
        'Median z-Score: <b>%{z:.3f}</b><br>' +
        'ELNs: <b>%{customdata[0]}</b>' +
        '<extra></extra>',
      hoverlabel: getHoverLabelStyle(isDark),
    },
  ];

  const bg = isDark ? 'rgba(0,0,0,0)' : '#fff';
  const axisColor = isDark ? '#aaa' : undefined;
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
      title: { text: xCol, font: { size: fontSize + 2, family: monoFont, color: axisColor }, standoff: 20 },
      tickangle: -45,
      tickfont: { size: presentationMode ? 16 : 13, family: monoFont, color: axisColor },
      side: 'bottom',
      automargin: true,
      showgrid: false,
    },
    yaxis: {
      title: { text: yCol, font: { size: fontSize + 2, family: monoFont, color: axisColor }, standoff: 20 },
      tickfont: { size: presentationMode ? 16 : 13, family: monoFont, color: axisColor },
      automargin: true,
      showgrid: false,
      ticksuffix: rankAnnotations.length > 0 ? RANK_BADGE_TICK_PAD : undefined,
    },
    height,
    margin: { t: 60, b: 180, l: 220, r: 30 },
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: {
      family: '"DM Sans", "Helvetica Neue", sans-serif',
      size: fontSize,
      color: isDark ? '#ddd' : undefined,
    },
    annotations: rankAnnotations.length > 0 ? rankAnnotations : undefined,
  };

  return { data, layout };
}
