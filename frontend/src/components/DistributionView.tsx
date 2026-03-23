import Plot from './Plot';
import { useFilterStore } from '../stores/filterStore';
import type { PlotConfig } from '../plots/types';
import type { Row } from '../data/types';

type ConfigBuilder = (rows: Row[], reactantTypes: string[], presentationMode: boolean) => PlotConfig;

interface Props {
  buildConfig: ConfigBuilder;
  label: string;
  rows: Row[];
  reactantTypes: string[];
}

export function DistributionView({ buildConfig, label, rows, reactantTypes }: Props) {
  const presentationMode = useFilterStore((s) => s.presentationMode);

  if (reactantTypes.length === 0) {
    return (
      <div className="plot-container">
        <p className="no-data-message">
          Select a reactant type to display the {label}.
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
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}
