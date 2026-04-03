/**
 * Shared helpers for distribution plot builders (boxplot, violin).
 *
 * Extracts grouping, sorting, hover text, median overlay, and layout logic
 * so that each plot type only needs to define its trace-specific properties.
 */

import type { Row, RankDelta, ComparisonInfo } from '../data/types';
import type { Data, Layout } from 'plotly.js';
import { createColorMappingFromElnCounts } from './colors';
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

/** Compute median of a numeric array (copies + sorts internally). */
export function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  return medianOfSorted(sorted);
}

/** Compute median of a pre-sorted numeric array (no copy, no sort). */
export function medianOfSorted(sorted: number[]): number {
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

/** Build a rich HTML hover tooltip for a rank badge annotation. */
export function rankBadgeHoverText(delta: RankDelta, info?: ComparisonInfo | null, isDark = false): string {
  const dim = isDark ? '#aaa' : '#999';
  const header =
    `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">COMPARISON</span><br>`;

  const versionLine = info
    ? `${info.comparisonLabel} → ${info.currentLabel}<br><br>`
    : '';

  if (delta.isNew) {
    return (
      header + versionLine +
      `<b style="color:#2196F3">NEW</b><br>` +
      `<span style="color:${dim}">Not in comparison dataset</span><br><br>` +
      `<span style="color:${dim}">Median:</span> <b>${delta.currentMedian.toFixed(3)}</b>`
    );
  }

  const rankColor = rankBadgeColor(delta);
  const badge = formatRankBadge(delta);
  const mSign = delta.medianDelta >= 0 ? '+' : '';

  return (
    header + versionLine +
    `<span style="color:${dim}">Rank:</span> ${delta.comparisonRank} → <b>${delta.currentRank}</b> ` +
    `<span style="color:${rankColor}"><b>${badge}</b></span><br>` +
    `<span style="color:${dim}">Median:</span> ${delta.comparisonMedian.toFixed(3)} → <b>${delta.currentMedian.toFixed(3)}</b> ` +
    `<span style="color:${dim}">(${mSign}${delta.medianDelta.toFixed(3)})</span>`
  );
}

/** Whitespace appended to y-axis tick labels to make room for rank badge annotations. */
export const RANK_BADGE_TICK_PAD = '        ';

/** Build a Plotly annotation object for a rank badge next to an axis label. */
export function buildRankAnnotation(
  delta: RankDelta,
  label: string,
  axis: 'y' | 'x-top',
  fontSize: number,
  comparisonInfo?: ComparisonInfo | null,
  isDark = false,
): Partial<Layout>['annotations'] extends (infer A)[] | undefined ? A : never {
  const positioning = axis === 'y'
    ? { x: 0, xref: 'paper' as const, xanchor: 'right' as const, xshift: -4, y: label, yref: 'y' as const, yanchor: 'middle' as const }
    : { y: 1, yref: 'paper' as const, yanchor: 'bottom' as const, yshift: 4, x: label, xref: 'x' as const, xanchor: 'center' as const };
  return {
    text: `<b>${formatRankBadge(delta)}</b>`,
    hovertext: rankBadgeHoverText(delta, comparisonInfo, isDark),
    hoverlabel: getHoverLabelStyle(isDark),
    ...positioning,
    showarrow: false,
    font: { size: fontSize, family: '"JetBrains Mono", monospace', color: rankBadgeColor(delta) },
  };
}

// ── Prepared data structures ──────────────────────────────────────────

export interface PreparedGroup {
  name: string;
  rows: Row[];
  zScores: number[];
  medianVal: number;
  color: string;
  elnCount: number;
  /** Per-point structured data for Plotly hover (passed as trace.customdata). */
  customdata: string[][];
  /** Plotly hovertemplate referencing customdata indices (set once per trace). */
  hovertemplate: string;
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
  comparisonInfo?: ComparisonInfo | null,
): PreparedData | null {
  if (rows.length === 0 || reactantTypes.length === 0) return null;

  const groupCol = reactantTypes.length === 1
    ? reactantTypes[0]
    : reactantTypes.join(' / ');

  // Single pass: group rows by category, collect ELN sets for color mapping
  const groupMap = new Map<string, Row[]>();
  const elnSets = new Map<string, Set<string>>();
  for (const row of rows) {
    const key = reactantTypes
      .map((col) => String(row[col] ?? '(no value)'))
      .join(' / ');
    const z = row['z-Score'];
    if (z === null || z === undefined || isNaN(z)) continue;
    if (!groupMap.has(key)) {
      groupMap.set(key, []);
      elnSets.set(key, new Set());
    }
    groupMap.get(key)!.push(row);
    if (row.ELN_ID) elnSets.get(key)!.add(row.ELN_ID);
  }

  // ELN counts from the sets we already collected
  const elnCounts = new Map<string, number>();
  for (const [key, elns] of elnSets) {
    elnCounts.set(key, elns.size);
  }

  // Sort groups by median z-Score descending (sort zScores in-place to avoid re-sort in median)
  const sorted = Array.from(groupMap.entries())
    .map(([name, groupRows]) => {
      const zScores = groupRows.map((r) => r['z-Score'] as number);
      zScores.sort((a, b) => a - b);
      return { name, rows: groupRows, zScores, medianVal: medianOfSorted(zScores) };
    })
    .sort((a, b) => b.medianVal - a.medianVal);

  // Reverse for Plotly y-axis (bottom-to-top)
  const categoryOrder = sorted.map((g) => g.name).reverse();

  // Color mapping from pre-computed ELN counts (no extra row iteration)
  const combined = reactantTypes.length > 1;
  const colorMap = createColorMappingFromElnCounts(reactantTypes[0], elnCounts, combined);

  // Hover template — built once with theme color, references customdata indices
  const dim = isDark ? '#aaa' : '#999';
  const hovertemplate =
    `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">EXPERIMENT</span><br>` +
    `<b style="font-size:15px">%{customdata[0]}</b>` +
    `<span style="color:${dim}"> · Plate %{customdata[1]} · %{customdata[2]}</span><br>` +
    `<br>` +
    `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">RESULTS</span><br>` +
    `z-Score: <b>%{customdata[3]}</b> · Area: %{customdata[4]}<br>` +
    `<br>` +
    `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">REAGENTS</span><br>` +
    `%{customdata[5]}` +
    `<br>` +
    `<span style="color:${dim};font-size:12px;letter-spacing:0.05em">REACTION</span><br>` +
    `%{customdata[6]} · <b>%{customdata[7]} ELNs</b>` +
    `<extra></extra>`;

  // Build prepared groups with customdata (cheap array construction per row)
  const groups: PreparedGroup[] = sorted.map(({ name, rows: groupRows, zScores, medianVal }) => {
    const color = colorMap.get(name) ?? '#999';
    const elnCount = elnCounts.get(name) ?? 0;
    const elnCountStr = String(elnCount);

    const customdata = groupRows.map((row) => {
      // Build reagent block — direct concatenation, no intermediate arrays
      let reagents = '';
      if (row.Catalyst) reagents += `<span style="color:${dim}">Catalyst:</span> ${row.Catalyst}<br>`;
      if (row.Solvent) reagents += `<span style="color:${dim}">Solvent:</span> ${row.Solvent}<br>`;
      if (row.Base) reagents += `<span style="color:${dim}">Base:</span> ${row.Base}<br>`;
      if (row.Ligand) reagents += `<span style="color:${dim}">Ligand:</span> ${row.Ligand}<br>`;
      if (row.Additive) reagents += `<span style="color:${dim}">Additive:</span> ${row.Additive}<br>`;
      if (row['Coupling Reagent']) reagents += `<span style="color:${dim}">Coupling Reagent:</span> ${row['Coupling Reagent']}<br>`;
      if (row['FG A']) reagents += `<span style="color:${dim}">FG A:</span> ${row['FG A']}<br>`;
      if (row['FG B']) reagents += `<span style="color:${dim}">FG B:</span> ${row['FG B']}<br>`;
      if (row['Secondary Solvent']) reagents += `<span style="color:${dim}">Secondary Solvent:</span> ${row['Secondary Solvent']}<br>`;

      return [
        s(row.ELN_ID),                  // 0
        s(row.PLATENUMBER),             // 1
        s(row.Coordinate),              // 2
        fmtZ(row['z-Score']),           // 3
        fmtArea(row.AREA_TOTAL_REDUCED),// 4
        reagents,                       // 5
        s(row['Reaction Type']),        // 6
        elnCountStr,                    // 7
      ];
    });

    return { name, rows: groupRows, zScores, medianVal, color, elnCount, customdata, hovertemplate };
  });

  // Build colored rank badge annotations (positioned next to y-axis labels)
  const rankAnnotations: Partial<Layout>['annotations'] = [];
  if (rankMap && rankMap.size > 0) {
    const badgeSize = presentationMode ? 20 : 15;
    for (const group of groups) {
      const delta = rankMap.get(group.name);
      if (delta) {
        rankAnnotations.push(buildRankAnnotation(delta, group.name, 'y', badgeSize, comparisonInfo, isDark));
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
export function buildMedianTrace(name: string, medianVal: number, n: number, elnCount: number, isDark = false): Data {
  return {
    type: 'scatter' as const,
    x: [medianVal],
    y: [name],
    mode: 'markers' as const,
    marker: { color: 'rgba(0,0,0,0)', size: 20 },
    showlegend: false,
    hovertemplate: `<b>${name}</b><br>Median: ${medianVal.toFixed(3)}<br>n = ${n} · ELNs: ${elnCount}<extra></extra>`,
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
  comparisonInfo?: ComparisonInfo | null,
): PlotConfig {
  const prepared = prepareDistributionData(rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo);
  if (!prepared) return { data: [], layout: {} };

  const data: Data[] = prepared.groups
    .map((group) => [buildTrace(group), buildMedianTrace(group.name, group.medianVal, group.zScores.length, group.elnCount, isDark)])
    .flat();

  return { data, layout: prepared.layout };
}
