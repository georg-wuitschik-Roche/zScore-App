/**
 * Boxplot configuration builder.
 *
 * Creates Plotly data + layout objects from filtered Row data.
 * Port of plot_utils.create_boxplot() from Python.
 */

import type { Row } from '../data/types';
import type { PlotConfig } from './types';
import { buildDistributionConfig, HOVER_LABEL_STYLE } from './helpers';

export type { PlotConfig };

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
    text: group.hoverText,
    hoverinfo: 'text' as const,
    hoveron: 'points' as const,
    hoverlabel: HOVER_LABEL_STYLE,
  }));
}
