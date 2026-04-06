/**
 * Boxplot configuration builder.
 *
 * Creates Plotly data + layout objects from filtered Row data.
 * Port of plot_utils.create_boxplot() from Python.
 */

import type { Row, RankDelta, ComparisonInfo } from '../data/types';
import type { PlotConfig } from './types';
import { buildDistributionConfig, getHoverLabelStyle } from './helpers';

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
  rankMap?: Map<string, RankDelta> | null,
  isDark = false,
  comparisonInfo?: ComparisonInfo | null,
  showElnLegend = true,
): PlotConfig {
  return buildDistributionConfig(rows, reactantTypes, presentationMode, (group) => ({
    type: 'box' as const,
    x: group.zScores,
    y: Array(group.zScores.length).fill(group.name),
    orientation: 'h' as const,
    name: group.name,
    boxpoints: 'all' as const,
    jitter: 0.3,
    pointpos: -1.5,
    boxmean: false,
    marker: { color: group.color, size: 6, opacity: 0.5 },
    line: { color: '#333', width: 1.5 },
    fillcolor: group.color,
    showlegend: false,
    customdata: group.customdata,
    hovertemplate: group.hovertemplate,
    hoveron: 'points' as const,
    hoverlabel: getHoverLabelStyle(isDark),
  }), rankMap, isDark, comparisonInfo, showElnLegend);
}
