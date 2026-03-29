/**
 * Shared helpers for distribution plot builders (boxplot, violin).
 *
 * Extracts grouping, sorting, hover text, median overlay, and layout logic
 * so that each plot type only needs to define its trace-specific properties.
 */

import type { Row, RankDelta } from '../data/types';
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

/** Format a rank delta as a text badge for plot labels. */
export function formatRankBadge(delta: RankDelta): string {
  if (delta.isNew) return 'NEW';
  if (delta.rankChange > 0) return `▲${delta.rankChange}`;
  if (delta.rankChange < 0) return `▼${Math.abs(delta.rankChange)}`;
  return '─';
}

/** Color for a rank delta badge. */
export function rankBadgeColor(delta: RankDelta): string {
  if (delta.isNew) return '#2196F3';
  if (delta.rankChange > 0) return '#4CAF50';
  if (delta.rankChange < 0) return '#F44336';
  return '#999';
}

/** Whitespace appended to y-axis tick labels to make room for rank badge annotations. */
export const RANK_BADGE_TICK_PAD = '        ';

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
  rankMap?: Map<string, RankDelta> | null,
  isDark = false,
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

  // Color mapping: ELN density → light/dark shade
  // Use slate palette when combining multiple reactant types, otherwise type-specific palette
  const combined = reactantTypes.length > 1;
  const colorMap = createColorMapping(reactantTypes[0], rows, reactantTypes, combined);

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

    const dim = isDark ? '#aaa' : '#999';
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
        .map(([k, v]) => `<span style="color:${dim}">${k}:</span> ${s(v)}`)
        .join('<br>');

      return (
        `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">EXPERIMENT</span><br>` +
        `<b style="font-size:15px">${s(row.ELN_ID)}</b>` +
        `<span style="color:${dim}"> · Plate ${s(row.PLATENUMBER)} · ${s(row.Coordinate)}</span><br>` +
        `<br>` +
        `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">RESULTS</span><br>` +
        `z-Score: <b>${fmtZ(row['z-Score'])}</b> · Area: ${fmtArea(row.AREA_TOTAL_REDUCED)}<br>` +
        `<br>` +
        `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">REAGENTS</span><br>` +
        reagentLines +
        `<br><br>` +
        `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">REACTION</span><br>` +
        `${s(row['Reaction Type'])} · <b>${elnCount} ELNs</b>`
      );
    });

    return { name, rows: groupRows, zScores, medianVal, color, elnCount, hoverText };
  });

  // Build colored rank badge annotations (positioned next to y-axis labels)
  const rankAnnotations: Partial<Layout>['annotations'] = [];
  if (rankMap && rankMap.size > 0) {
    for (const group of groups) {
      const delta = rankMap.get(group.name);
      if (delta) {
        rankAnnotations.push({
          text: `<b>${formatRankBadge(delta)}</b>`,
          x: 0,
          xref: 'paper',
          xanchor: 'right',
          xshift: -4,
          y: group.name,
          yref: 'y',
          yanchor: 'middle',
          showarrow: false,
          font: {
            size: presentationMode ? 20 : 15,
            family: '"JetBrains Mono", monospace',
            color: rankBadgeColor(delta),
          },
        });
      }
    }
  }

  // Layout
  const fontSize = presentationMode ? 18 : 14;
  const numCategories = sorted.length;
  const height = Math.max(800, numCategories * 110);

  const bg = isDark ? 'rgba(0,0,0,0)' : '#fff';
  const axisColor = isDark ? '#aaa' : undefined;
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
          color: axisColor,
        },
      },
      zeroline: true,
      zerolinecolor: isDark ? '#555' : '#ccc',
      gridcolor: isDark ? '#333' : '#d0d0d0',
      showgrid: true,
      tickfont: { size: presentationMode ? 16 : 13, family: '"JetBrains Mono", "Fira Code", monospace', color: axisColor },
    },
    yaxis: {
      automargin: true,
      categoryorder: 'array',
      categoryarray: categoryOrder,
      showgrid: false,
      tickfont: { size: presentationMode ? 20 : 14, family: '"JetBrains Mono", "Fira Code", monospace', color: axisColor },
      ticksuffix: rankAnnotations.length > 0 ? RANK_BADGE_TICK_PAD : undefined,
    },
    height,
    showlegend: false,
    margin: { t: 60, b: 80, l: 200, r: 50 },
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: {
      family: '"DM Sans", "Helvetica Neue", sans-serif',
      size: fontSize,
      color: isDark ? '#ddd' : undefined,
    },
    annotations: rankAnnotations.length > 0 ? rankAnnotations : undefined,
  };

  return { groups, categoryOrder, layout };
}

// ── Shared trace builders ─────────────────────────────────────────────

/** Invisible median marker — shows clean tooltip on distribution hover */
export function buildMedianTrace(name: string, medianVal: number, n: number, isDark = false): Data {
  return {
    type: 'scatter' as const,
    x: [medianVal],
    y: [name],
    mode: 'markers' as const,
    marker: { color: 'rgba(0,0,0,0)', size: 20 },
    showlegend: false,
    hovertemplate: `<b>${name}</b><br>Median: ${medianVal.toFixed(3)}<br>n = ${n}<extra></extra>`,
    hoverlabel: getHoverLabelStyle(isDark),
  };
}

/** Shared hover label style for distribution traces */
export function getHoverLabelStyle(isDark = false) {
  return {
    bgcolor: isDark ? '#1e1e1e' : '#fff',
    bordercolor: isDark ? '#444' : '#e0e0e0',
    font: { size: 14, family: '"JetBrains Mono", "Fira Code", monospace', color: isDark ? '#ddd' : '#222' },
    align: 'left' as const,
  };
}

/** Build a PlotConfig from prepared data, using a trace builder for each group */
export function buildDistributionConfig(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
  buildTrace: (group: PreparedGroup) => Data,
  rankMap?: Map<string, RankDelta> | null,
  isDark = false,
): PlotConfig {
  const prepared = prepareDistributionData(rows, reactantTypes, presentationMode, rankMap, isDark);
  if (!prepared) return { data: [], layout: {} };

  const data: Data[] = prepared.groups
    .map((group) => [buildTrace(group), buildMedianTrace(group.name, group.medianVal, group.zScores.length, isDark)])
    .flat();

  return { data, layout: prepared.layout };
}
