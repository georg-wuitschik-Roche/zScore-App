import { memo, useEffect, useMemo } from 'react';
import Plot from './Plot';
import { useFilterStore } from '../stores/filterStore';
import { createHeatmapConfig } from '../plots/heatmap';
import { useZoomReset } from './DistributionView';
import type { Row, RankDelta, ComparisonInfo } from '../data/types';

const PLOT_CONFIG = { responsive: true, displayModeBar: false } as const;
const PLOT_STYLE = { width: '100%' } as const;

interface Props {
  rows: Row[];
  reactantTypes: string[];
  noDataHint?: string;
  axisRankMaps?: Map<string, RankDelta>[] | null;
  comparisonInfo?: ComparisonInfo;
}

export const HeatmapView = memo(function HeatmapView({ rows, reactantTypes, noDataHint, axisRankMaps, comparisonInfo }: Props) {
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const isDark = useFilterStore((s) => s.theme) === 'dark';

  const { isZoomed, setIsZoomed, handleInit, resetZoom } = useZoomReset();

  const config = useMemo(
    () => rows.length > 0 && reactantTypes.length >= 2
      ? createHeatmapConfig(rows, reactantTypes, presentationMode, axisRankMaps, isDark, comparisonInfo)
      : null,
    [rows, reactantTypes, presentationMode, axisRankMaps, isDark, comparisonInfo],
  );

  useEffect(() => { setIsZoomed(false); }, [config, setIsZoomed]);

  if (reactionTypes.length === 0 || reactantTypes.length < 2) {
    const missing: string[] = [];
    if (reactionTypes.length === 0) missing.push('reaction type');
    if (reactantTypes.length < 2) missing.push('at least 2 reactant types');
    return (
      <div className="plot-container empty-state">
        <img src="/assets/logo.svg" alt="" className="empty-state-logo" />
        <p className="no-data-message">
          Select {missing.join(' and ')} for heatmap view.
        </p>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="plot-container">
        <p className="no-data-message">
          {noDataHint ?? 'No data available for the current filter selection.'}
        </p>
      </div>
    );
  }

  return (
    <div className="plot-container plot-container--zoomable">
      {isZoomed && (
        <button className="reset-zoom-btn" onClick={resetZoom} title="Reset zoom">
          Reset Zoom
        </button>
      )}
      <Plot
        data={config.data}
        layout={config.layout}
        config={PLOT_CONFIG}
        style={PLOT_STYLE}
        useResizeHandler
        onInitialized={handleInit}
      />
    </div>
  );
});
