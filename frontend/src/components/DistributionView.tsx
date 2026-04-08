import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Plot, { Plotly } from './Plot';
import { useFilterStore } from '../stores/filterStore';
import type { PlotConfig } from '../plots/types';
import type { Row, RankDelta, ComparisonInfo } from '../data/types';

const PLOT_CONFIG = { responsive: true, displayModeBar: false } as const;
const PLOT_STYLE = { width: '100%' } as const;

export function useZoomReset() {
  const [isZoomed, setIsZoomed] = useState(false);
  const plotDivRef = useRef<ReturnType<typeof Plotly.newPlot> extends Promise<infer R> ? R : unknown>(null);

  const handleInit = useCallback((_figure: unknown, graphDiv: HTMLElement) => {
    plotDivRef.current = graphDiv;
    (graphDiv as unknown as { on: (e: string, h: (d: Record<string, unknown>) => void) => void }).on(
      'plotly_relayout',
      (data: Record<string, unknown>) => {
        const keys = Object.keys(data);
        if (keys.some(k => /[xy]axis\d*\.range/.test(k))) setIsZoomed(true);
        else if (keys.some(k => /[xy]axis\d*\.autorange/.test(k))) setIsZoomed(false);
      },
    );
  }, []);

  const resetZoom = useCallback(() => {
    if (plotDivRef.current) {
      Plotly.relayout(plotDivRef.current, { 'xaxis.autorange': true, 'yaxis.autorange': true });
    }
  }, []);

  return { isZoomed, setIsZoomed, handleInit, resetZoom };
}

type ConfigBuilder = (
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
  rankMap?: Map<string, RankDelta> | null,
  isDark?: boolean,
  comparisonInfo?: ComparisonInfo | null,
  showElnLegend?: boolean,
) => PlotConfig;

interface Props {
  buildConfig: ConfigBuilder;
  label: string;
  rows: Row[];
  reactantTypes: string[];
  noDataHint?: string;
  rankMap?: Map<string, RankDelta> | null;
  comparisonInfo?: ComparisonInfo | null;
  heightOverride?: number;
}

export const DistributionView = memo(function DistributionView({ buildConfig, label, rows, reactantTypes, noDataHint, rankMap, comparisonInfo, heightOverride }: Props) {
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const isDark = useFilterStore((s) => s.theme) === 'dark';
  const showElnLegend = useFilterStore((s) => s.showElnLegend);

  const { isZoomed, setIsZoomed, handleInit, resetZoom } = useZoomReset();

  const config = useMemo(() => {
    if (rows.length === 0) return null;
    const c = buildConfig(rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo, showElnLegend);
    if (heightOverride && c.layout) {
      c.layout.height = heightOverride;
    }
    return c;
  }, [buildConfig, rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo, showElnLegend, heightOverride]);

  useEffect(() => { setIsZoomed(false); }, [config, setIsZoomed]);

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
    <div className="plot-container plot-container--zoomable">
      {isZoomed && (
        <button className="reset-zoom-btn" onClick={resetZoom} title="Reset zoom">
          Reset Zoom
        </button>
      )}
      <Plot
        key={showElnLegend ? 'legend' : 'no-legend'}
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
