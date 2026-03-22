import Plot from './Plot';
import { useFilteredData } from '../hooks/useFilteredData';
import { useFilterStore } from '../stores/filterStore';
import { createHeatmapConfig } from '../plots/heatmap';
import type { Row } from '../data/types';

interface Props {
  /** When provided, bypasses useFilteredData (used by split mode). */
  panelRows?: Row[];
  /** When provided, overrides store reactantTypes (used by split mode). */
  panelReactantTypes?: string[];
}

export function HeatmapView({ panelRows, panelReactantTypes }: Props = {}) {
  const fallback = useFilteredData();
  const storeReactantTypes = useFilterStore((s) => s.reactantTypes);
  const presentationMode = useFilterStore((s) => s.presentationMode);

  const rows = panelRows ?? fallback.rows;
  const reactantTypes = panelReactantTypes ?? storeReactantTypes;

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
          No data available for the current filter selection.
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
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
