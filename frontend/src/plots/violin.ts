/**
 * Violin plot configuration builder.
 *
 * Creates Plotly data + layout objects from filtered Row data.
 * Uses violin traces for kernel density visualization.
 */

import type { Row, RankDelta, ComparisonInfo } from '../data/types';
import type { PlotConfig } from './types';
import { prepareDistributionData, buildMedianTrace, getHoverLabelStyle } from './helpers';
import type { Data } from 'plotly.js';

/** Silverman's rule-of-thumb bandwidth (matches Plotly's default KDE) */
function silvermanBandwidth(data: number[]): number {
  const n = data.length;
  if (n < 2) return 1;
  const mean = data.reduce((s, x) => s + x, 0) / n;
  const std = Math.sqrt(data.reduce((s, x) => s + (x - mean) ** 2, 0) / n);
  const sorted = [...data].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const spread = Math.min(std, iqr / 1.34);
  return 1.06 * (spread > 0 ? spread : std > 0 ? std : 1) * Math.pow(n, -0.2);
}

/** Evaluate Gaussian KDE at a single point */
function kdeAt(data: number[], bandwidth: number, x: number): number {
  let sum = 0;
  for (const xi of data) {
    const z = (x - xi) / bandwidth;
    sum += Math.exp(-0.5 * z * z);
  }
  return sum / (data.length * bandwidth);
}

/** Find the violin half-width at a given x, as a fraction of the max width (0..1) */
function violinFractionAt(data: number[], x: number): number {
  const bw = silvermanBandwidth(data);
  const atX = kdeAt(data, bw, x);
  // Sample KDE to find max — evaluate at each data point + 50 evenly spaced points
  const min = Math.min(...data);
  const max = Math.max(...data);
  let maxKde = atX;
  const steps = 50;
  for (let i = 0; i <= steps; i++) {
    const xi = min + (max - min) * (i / steps);
    maxKde = Math.max(maxKde, kdeAt(data, bw, xi));
  }
  return maxKde > 0 ? atX / maxKde : 0;
}

/**
 * Build Plotly violin config from filtered rows.
 *
 * Groups z-Score values by the selected reactant type(s) and creates
 * one violin trace per unique category value, sorted by descending median.
 * Each data point has a rich hover tooltip with experiment details.
 */
export function createViolinConfig(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean = false,
  rankMap?: Map<string, RankDelta> | null,
  isDark = false,
  comparisonInfo?: ComparisonInfo | null,
  showElnLegend = true,
): PlotConfig {
  const prepared = prepareDistributionData(rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo, showElnLegend);
  if (!prepared) return { data: [], layout: {} };

  const data: Data[] = prepared.groups
    .map((group) => [
      {
        type: 'violin' as const,
        x: group.zScores,
        y: Array(group.zScores.length).fill(group.name),
        orientation: 'h' as const,
        name: group.name,
        points: 'all' as const,
        jitter: 0.3,
        pointpos: -1.5,
        marker: { color: group.color, size: 6, opacity: 0.5 },
        line: { color: '#333', width: 1.5 },
        fillcolor: group.color,
        showlegend: false,
        customdata: group.customdata,
        hovertemplate: group.hovertemplate,
        hoveron: 'points' as const,
        hoverlabel: getHoverLabelStyle(isDark),
      } as Data,
      buildMedianTrace(group.name, group.medianVal, group.zScores.length, group.elnCount, isDark),
    ])
    .flat();
  if (prepared.colorbarTrace) data.push(prepared.colorbarTrace);

  // Dashed median lines bounded by violin outline
  const MAX_HALF_WIDTH = 0.4; // Plotly's default max half-width per category
  const shapes = prepared.groups.map((group) => {
    const idx = prepared.categoryOrder.indexOf(group.name);
    const fraction = violinFractionAt(group.zScores, group.medianVal);
    const halfWidth = fraction * MAX_HALF_WIDTH;
    return {
      type: 'line' as const,
      x0: group.medianVal,
      x1: group.medianVal,
      y0: idx - halfWidth,
      y1: idx + halfWidth,
      yref: 'y' as const,
      xref: 'x' as const,
      line: { color: '#333', width: 1.5, dash: 'dash' as const },
    };
  });

  return {
    data,
    layout: {
      ...prepared.layout,
      shapes,
    },
  };
}
