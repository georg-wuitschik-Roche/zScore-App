import Plot from './Plot';
import { useFilteredData } from '../hooks/useFilteredData';
import { useFilterStore } from '../stores/filterStore';
import type { PlotConfig } from '../plots/types';
import type { Row } from '../data/types';

type ConfigBuilder = (rows: Row[], reactantTypes: string[], presentationMode: boolean) => PlotConfig;

interface Props {
  buildConfig: ConfigBuilder;
  label: string;
  /** When provided, bypasses useFilteredData (used by split mode). */
  panelRows?: Row[];
  /** When provided, overrides store reactantTypes (used by split mode). */
  panelReactantTypes?: string[];
}

export function DistributionView({ buildConfig, label, panelRows, panelReactantTypes }: Props) {
  const fallback = useFilteredData();
  const storeReactantTypes = useFilterStore((s) => s.reactantTypes);
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);

  const rows = panelRows ?? fallback.rows;
  const reactantTypes = panelReactantTypes ?? storeReactantTypes;

  if (reactionTypes.length === 0 || reactantTypes.length === 0) {
    return (
      <div className="plot-container">
        <p className="no-data-message">
          Select a reaction type and reactant type to display the {label}.
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

  const config = buildConfig(rows, reactantTypes, presentationMode);

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
