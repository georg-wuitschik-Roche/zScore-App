/**
 * Shared helpers for distribution plot builders (boxplot, violin).
 *
 * Extracts grouping, sorting, hover payloads, median overlay, and layout logic
 * so that each plot type only needs to define its trace-specific geometry.
 */

import type { Row, RankDelta, ComparisonInfo } from '../data/types';
import type { Data, Layout } from 'plotly.js';
import { createColorMappingFromElnCounts, BASE_COLOURS, COMBINED_COLOURS, DEFAULT_COLOURS } from './colors';
import type { PlotConfig } from './types';
import { asCustomdata, type TooltipSource } from './tooltip';

// ── Math helpers ──────────────────────────────────────────────────────

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
    ? `<span style="color:${dim}">Viewing:</span> <b>${info.currentLabel}</b><br>` +
      `<span style="color:${dim}">Baseline:</span> ${info.comparisonLabel}<br><br>`
    : '';

  if (delta.isNew) {
    return (
      header + versionLine +
      `<b style="color:#2196F3">NEW</b><br>` +
      `<span style="color:${dim}">Not in baseline dataset</span><br><br>` +
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

// ── ELN density colorbar ─────────────────────────────────────────────

export const MONO_FONT = '"JetBrains Mono", "Fira Code", monospace';

/** "ELN Count" label annotation positioned to the left of the colorbar. */
function buildElnLabel(presentationMode: boolean, axisColor?: string) {
  return {
    text: '<b>ELN Count</b>',
    xref: 'paper' as const,
    yref: 'paper' as const,
    x: 0,
    xanchor: 'right' as const,
    xshift: -8,
    y: 1,
    yanchor: 'bottom' as const,
    yshift: 4,
    showarrow: false,
    font: { size: presentationMode ? 14 : 11, family: MONO_FONT, color: axisColor ?? '#999' },
  };
}

/** Invisible scatter trace with a colorbar showing the ELN density gradient. */
function buildElnColorbar(
  category: string,
  combined: boolean,
  groups: { elnCount: number }[],
  presentationMode: boolean,
  axisColor?: string,
): Data {
  const base = combined ? COMBINED_COLOURS : (BASE_COLOURS[category] ?? DEFAULT_COLOURS);
  const elnValues = groups.map((g) => g.elnCount);
  const minElns = Math.min(...elnValues);
  const maxElns = Math.max(...elnValues);
  const range = maxElns - minElns;
  return {
    type: 'scatter' as const,
    x: [null],
    y: [null],
    mode: 'markers' as const,
    marker: {
      color: [minElns, maxElns],
      colorscale: [[0, base.light], [1, base.dark]],
      cmin: minElns,
      cmax: maxElns,
      showscale: true,
      colorbar: {
        title: undefined,
        tick0: 0,
        dtick: Math.max(1, Math.ceil(range / 4 / 5) * 5) || 5,
        tickfont: { size: presentationMode ? 13 : 10, family: MONO_FONT, color: axisColor },
        thickness: 10,
        lenmode: 'fraction' as const,
        len: 1,
        orientation: 'h' as const,
        yref: 'paper' as const,
        y: 1,
        yanchor: 'bottom' as const,
        ypad: 2,
        outlinewidth: 0,
      },
      size: 0.01,
      opacity: 0,
    },
    showlegend: false,
    hoverinfo: 'skip' as const,
  };
}

// ── Tick label wrapping ──────────────────────────────────────────────

/**
 * Wrap a long tick label by inserting `<br>` at natural break points
 * so Plotly renders it as multi-line, saving horizontal space.
 */
export function wrapTickLabel(label: string, maxLen = 18): string {
  if (label.length <= maxLen) return label;

  // Split at spaces (covers compound keys like "CatA / BaseB" too)
  const parts = label.split(' ');
  if (parts.length > 1) {
    const lines: string[] = [];
    let current = parts[0];
    for (let i = 1; i < parts.length; i++) {
      if ((current + ' ' + parts[i]).length <= maxLen) {
        current += ' ' + parts[i];
      } else {
        lines.push(current);
        current = parts[i];
      }
    }
    lines.push(current);
    return lines.join('<br>');
  }

  // No spaces — break after closing bracket nearest to the middle
  const breakPattern = /[\])]/g;
  let bestBreak = -1;
  let match;
  while ((match = breakPattern.exec(label)) !== null) {
    const pos = match.index + 1;
    if (pos >= 4 && pos <= label.length - 2) {
      bestBreak = pos;
      if (pos >= maxLen * 0.5) break;
    }
  }
  if (bestBreak > 0) {
    return label.slice(0, bestBreak) + '<br>' + label.slice(bestBreak);
  }

  // Last resort: hard break
  return label.slice(0, maxLen) + '<br>' + label.slice(maxLen);
}

// ── Prepared data structures ──────────────────────────────────────────

export interface PreparedGroup {
  name: string;
  zScores: number[];
  medianVal: number;
  color: string;
  elnCount: number;
  /** Per-point hover payload, index-aligned with `zScores` (passed as trace.customdata). */
  customdata: TooltipSource[];
}

export interface PreparedData {
  groups: PreparedGroup[];
  categoryOrder: string[];
  layout: Partial<Layout>;
  /** Invisible scatter trace that renders a colorbar showing the ELN density gradient (null when hidden). */
  colorbarTrace: Data | null;
}

// ── Core preparation logic ────────────────────────────────────────────

/**
 * Prepare all shared data for a distribution plot.
 *
 * Groups rows by reactant types (compound key when multiple are selected),
 * sorts by descending median, computes colors/hover payloads/layout.
 * Returns null for empty inputs.
 */
export function prepareDistributionData(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
  rankMap?: Map<string, RankDelta> | null,
  isDark = false,
  comparisonInfo?: ComparisonInfo | null,
  showElnLegend = true,
): PreparedData | null {
  if (rows.length === 0 || reactantTypes.length === 0) return null;

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

  // Sort groups by median z-Score descending.
  // Rows are sorted by z-Score first so that zScores and customdata stay index-aligned:
  // Plotly maps a hovered box/violin point back to its original input index, so a point's
  // x value and its hover payload must come from the same position in the array.
  const sorted = Array.from(groupMap.entries())
    .map(([name, groupRows]) => {
      const rows = [...groupRows].sort(
        (a, b) => (a['z-Score'] as number) - (b['z-Score'] as number),
      );
      const zScores = rows.map((r) => r['z-Score'] as number);
      return { name, rows, zScores, medianVal: medianOfSorted(zScores) };
    })
    .sort((a, b) => b.medianVal - a.medianVal || a.name.localeCompare(b.name));

  // Reverse for Plotly y-axis (bottom-to-top)
  const categoryOrder = sorted.map((g) => g.name).reverse();

  // Color mapping from pre-computed ELN counts (no extra row iteration)
  const combined = reactantTypes.length > 1;
  const colorMap = createColorMappingFromElnCounts(reactantTypes[0], elnCounts, combined);

  // Build prepared groups with hover payloads (one small object per point — the
  // display model is built lazily on hover, not here in the filter-hot path)
  const groups: PreparedGroup[] = sorted.map(({ name, rows: groupRows, zScores, medianVal }) => {
    const color = colorMap.get(name) ?? '#999';
    const elnCount = elnCounts.get(name) ?? 0;

    const customdata: TooltipSource[] = groupRows.map((row) => ({
      kind: 'point',
      row,
      elnCount,
    }));

    return { name, zScores, medianVal, color, elnCount, customdata };
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
  const wrappedLabels = categoryOrder.map((label) => wrapTickLabel(label));
  const hasRankBadges = rankAnnotations.length > 0;
  // Bake rank badge padding into ticktext (ticksuffix is ignored with custom ticktext)
  const tickLabels = hasRankBadges
    ? wrappedLabels.map((l) =>
        l.split('<br>').map((line) => line + RANK_BADGE_TICK_PAD).join('<br>'))
    : wrappedLabels;
  const height = Math.max(800, numCategories * 110);

  const bg = isDark ? 'rgba(0,0,0,0)' : '#fff';
  const axisColor = isDark ? '#aaa' : undefined;
  const layout: Partial<Layout> = {
    title: { text: '' },
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
      tickvals: categoryOrder,
      ticktext: tickLabels,
      showgrid: false,
      tickfont: { size: presentationMode ? 20 : 14, family: '"JetBrains Mono", "Fira Code", monospace', color: axisColor },
      domain: showElnLegend ? [0, 0.97] : undefined,
    },
    height,
    showlegend: false,
    margin: { t: showElnLegend ? 32 : 4, b: 80, l: 200, r: 50 },
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: {
      family: '"DM Sans", "Helvetica Neue", sans-serif',
      size: fontSize,
      color: isDark ? '#ddd' : undefined,
    },
    annotations: rankAnnotations.length > 0 ? rankAnnotations : undefined,
  };

  // ELN density colorbar + label (conditional on showElnLegend)
  let colorbarTrace: Data | null = null;
  if (showElnLegend) {
    colorbarTrace = buildElnColorbar(reactantTypes[0], combined, groups, presentationMode, axisColor);
    layout.annotations = [...(layout.annotations ?? []), buildElnLabel(presentationMode, axisColor)];
  }

  return { groups, categoryOrder, layout, colorbarTrace };
}

// ── Shared trace builders ─────────────────────────────────────────────

/** Invisible median marker — shows clean tooltip on distribution hover */
export function buildMedianTrace(name: string, medianVal: number, n: number, elnCount: number): Data {
  return {
    type: 'scatter' as const,
    x: [medianVal],
    y: [name],
    mode: 'markers' as const,
    marker: { color: 'rgba(0,0,0,0)', size: 20 },
    showlegend: false,
    // 'none' (not 'skip') keeps the trace in the hover search so plotly_hover
    // still fires for the custom tooltip, while suppressing the native label.
    hoverinfo: 'none' as const,
    customdata: asCustomdata([{ kind: 'median', category: name, medianVal, n, elnCount }]),
  };
}

/** Hover label style for the rank badge annotations, which keep Plotly's native label. */
function getHoverLabelStyle(isDark = false) {
  return {
    bgcolor: isDark ? '#1e1e1e' : '#fff',
    bordercolor: isDark ? '#444' : '#e0e0e0',
    font: { size: 14, family: '"JetBrains Mono", "Fira Code", monospace', color: isDark ? '#ddd' : '#222' },
    align: 'left' as const,
  };
}

/** Build a PlotConfig from prepared data, using a trace builder for each group.
 *  `buildTrace` supplies geometry only — the hover contract is applied here so no
 *  trace builder can forget it. An optional layoutModifier receives the prepared
 *  data to add plot-type-specific layout properties (e.g. violin median lines). */
export function buildDistributionConfig(
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
  buildTrace: (group: PreparedGroup) => Data,
  rankMap?: Map<string, RankDelta> | null,
  isDark = false,
  comparisonInfo?: ComparisonInfo | null,
  showElnLegend = true,
  layoutModifier?: (prepared: PreparedData) => Partial<Layout>,
): PlotConfig {
  const prepared = prepareDistributionData(rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo, showElnLegend);
  if (!prepared) return { data: [], layout: {} };

  const data: Data[] = prepared.groups
    .map((group) => [
      {
        ...buildTrace(group),
        // 'none' (not 'skip') keeps the trace in the hover search so plotly_hover
        // still fires for the custom tooltip, while suppressing the native label.
        hoverinfo: 'none' as const,
        customdata: asCustomdata(group.customdata),
      },
      buildMedianTrace(group.name, group.medianVal, group.zScores.length, group.elnCount),
    ])
    .flat();
  if (prepared.colorbarTrace) data.push(prepared.colorbarTrace);

  const layout = layoutModifier
    ? { ...prepared.layout, ...layoutModifier(prepared) }
    : prepared.layout;

  return { data, layout };
}
