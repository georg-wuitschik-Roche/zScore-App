/**
 * Shared types for plot configuration builders.
 */

import type { Data, Layout } from 'plotly.js';

export interface PlotConfig {
  data: Data[];
  layout: Partial<Layout>;
}
