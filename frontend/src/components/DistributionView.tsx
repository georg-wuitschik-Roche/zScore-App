import { memo, useMemo } from 'react';
import Plot from './Plot';
import { useFilterStore } from '../stores/filterStore';
import type { PlotConfig } from '../plots/types';
import type { Row, RankDelta, ComparisonInfo } from '../data/types';

type ConfigBuilder = (
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
  rankMap?: Map<string, RankDelta> | null,
  isDark?: boolean,
  comparisonInfo?: ComparisonInfo | null,
) => PlotConfig;

interface Props {
  buildConfig: ConfigBuilder;
  label: string;
  rows: Row[];
  reactantTypes: string[];
  noDataHint?: string;
  rankMap?: Map<string, RankDelta> | null;
  comparisonInfo?: ComparisonInfo;
}

export const DistributionView = memo(function DistributionView({ buildConfig, label, rows, reactantTypes, noDataHint, rankMap, comparisonInfo }: Props) {
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const isDark = useFilterStore((s) => s.theme) === 'dark';

  const config = useMemo(
    () => rows.length > 0 ? buildConfig(rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo) : null,
    [buildConfig, rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo],
  );

  if (reactionTypes.length === 0 || reactantTypes.length === 0) {
    const missing: string[] = [];
    if (reactionTypes.length === 0) missing.push('reaction type');
    if (reactantTypes.length === 0) missing.push('reactant type');
    return (
      <div className="plot-container empty-state">
        <img src="/assets/logo.svg" alt="" className="empty-state-logo" />
        <p className="no-data-message">
          Select a {missing.join(' and ')} to display the {label}.
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
});
