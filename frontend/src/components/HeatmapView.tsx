import Plot from './Plot';
import { useFilterStore } from '../stores/filterStore';
import { createHeatmapConfig } from '../plots/heatmap';
import type { Row } from '../data/types';

interface Props {
  rows: Row[];
  reactantTypes: string[];
  noDataHint?: string;
}

export function HeatmapView({ rows, reactantTypes, noDataHint }: Props) {
  const presentationMode = useFilterStore((s) => s.presentationMode);

  if (reactantTypes.length < 2) {
    return (
      <div className="plot-container">
        <p className="no-data-message">
          Select at least 2 reactant types for heatmap view.
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

  const config = createHeatmapConfig(rows, reactantTypes, presentationMode);

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
