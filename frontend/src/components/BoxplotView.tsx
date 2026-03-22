import Plot from './Plot';
import { useFilteredData } from '../hooks/useFilteredData';
import { useFilterStore } from '../stores/filterStore';
import { createBoxplotConfig } from '../plots/boxplot';

export function BoxplotView() {
  const { rows } = useFilteredData();
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const presentationMode = useFilterStore((s) => s.presentationMode);

  const reactionTypes = useFilterStore((s) => s.reactionTypes);

  if (reactionTypes.length === 0 || reactantTypes.length === 0) {
    return (
      <div className="plot-container">
        <p className="no-data-message">
          Select a reaction type and reactant type to display the boxplot.
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

  const config = createBoxplotConfig(rows, reactantTypes, presentationMode);

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
