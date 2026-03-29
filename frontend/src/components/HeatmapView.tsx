import Plot from './Plot';
import { useFilterStore } from '../stores/filterStore';
import { createHeatmapConfig } from '../plots/heatmap';
import type { Row, RankDelta } from '../data/types';

interface Props {
  rows: Row[];
  reactantTypes: string[];
  noDataHint?: string;
  axisRankMaps?: Map<string, RankDelta>[] | null;
}

export function HeatmapView({ rows, reactantTypes, noDataHint, axisRankMaps }: Props) {
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const isDark = useFilterStore((s) => s.theme) === 'dark';

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

  if (rows.length === 0) {
    return (
      <div className="plot-container">
        <p className="no-data-message">
          {noDataHint ?? 'No data available for the current filter selection.'}
        </p>
      </div>
    );
  }

  const config = createHeatmapConfig(rows, reactantTypes, presentationMode, axisRankMaps, isDark);

  return (
    <div className="plot-container">
      <Plot
        data={config.data}
        layout={config.layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
