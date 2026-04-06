import { memo, useDeferredValue, useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { DistributionView } from './DistributionView';
import { createBoxplotConfig } from '../plots/boxplot';
import { createViolinConfig } from '../plots/violin';
import { HeatmapView } from './HeatmapView';
import { StatsTable } from './StatsTable';
import { useSplitFilteredData } from '../hooks/useSplitFilteredData';
import { useComparisonRawData, useComparisonRanks } from '../hooks/useComparisonData';
import type { ComparisonResult } from '../hooks/useComparisonData';
import type { TabId, SplitPanel } from '../data/types';

interface TabDef {
  id: TabId;
  label: string;
  requiresMultiReactant: boolean;
}

const TABS: TabDef[] = [
  { id: 'violin', label: 'Violin', requiresMultiReactant: false },
  { id: 'boxplot', label: 'Boxplot', requiresMultiReactant: false },
  { id: 'heatmap', label: 'Heatmap', requiresMultiReactant: true },
  { id: 'stats', label: 'Stats', requiresMultiReactant: false },
];

function renderPanel(tab: TabId, panel: SplitPanel, comparison: ComparisonResult | null, heightOverride?: number) {
  const { rows, reactantTypes, stats } = panel;
  const noDataHint = stats.noDataHint;
  const comparisonInfo = comparison?.info;
  switch (tab) {
    case 'boxplot':
      return (
        <DistributionView
          buildConfig={createBoxplotConfig}
          label="boxplot"
          rows={rows}
          reactantTypes={reactantTypes}
          noDataHint={noDataHint}
          rankMap={comparison?.rankMap}
          comparisonInfo={comparisonInfo}
          heightOverride={heightOverride}
        />
      );
    case 'violin':
      return (
        <DistributionView
          buildConfig={createViolinConfig}
          label="violin plot"
          rows={rows}
          reactantTypes={reactantTypes}
          noDataHint={noDataHint}
          rankMap={comparison?.rankMap}
          comparisonInfo={comparisonInfo}
          heightOverride={heightOverride}
        />
      );
    case 'heatmap':
      return (
        <HeatmapView
          rows={rows}
          reactantTypes={reactantTypes}
          noDataHint={noDataHint}
          axisRankMaps={comparison?.axisRankMaps}
          comparisonInfo={comparisonInfo}
        />
      );
    case 'stats':
      return (
        <StatsTable
          rows={rows}
          reactantTypes={reactantTypes}
          noDataHint={noDataHint}
          rankMap={comparison?.rankMap}
        />
      );
  }
}

/** Wrapper that filters comparison data per-panel and computes rank deltas. */
const PanelWithComparison = memo(function PanelWithComparison({
  tab,
  panel,
  comparisonRawData,
  heightOverride,
}: {
  tab: TabId;
  panel: SplitPanel;
  comparisonRawData: ReturnType<typeof useComparisonRawData>;
  heightOverride?: number;
}) {
  const comparison = useComparisonRanks(panel.rows, panel.reactantTypes, comparisonRawData);
  return renderPanel(tab, panel, comparison, heightOverride);
});

export function AnalysisTabs() {
  const activeTab = useFilterStore((s) => s.activeTab);
  const setActiveTab = useFilterStore((s) => s.setActiveTab);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const splitSelector = useFilterStore((s) => s.splitSelector);
  const panels = useSplitFilteredData();

  // Defer panel data so tab buttons and UI stay responsive while charts recompute
  const deferredPanels = useDeferredValue(panels);

  // Resolve comparison version + raw rows (filtering deferred to per-panel)
  const comparisonRawData = useComparisonRawData();

  const isSplit = deferredPanels.length > 1;

  // Heatmap needs ≥2 reactant types per panel; when splitting by reactant types
  // each panel has only 1, so hide heatmap
  const showHeatmap =
    reactantTypes.length >= 2 && splitSelector !== 'reactantTypes';

  // If user was on heatmap and it becomes unavailable, fall back to boxplot
  const effectiveTab =
    activeTab === 'heatmap' && !showHeatmap ? 'boxplot' : activeTab;

  const visibleTabs = TABS.filter(
    (tab) => !tab.requiresMultiReactant || showHeatmap,
  );

  const showElnLegend = useFilterStore((s) => s.showElnLegend);
  const toggleElnLegend = useFilterStore((s) => s.toggleElnLegend);

  const isDistributionTab = effectiveTab === 'violin' || effectiveTab === 'boxplot';

  // Compute shared plot height so all split panels align
  const splitHeight = useMemo(() => {
    if (!isSplit || !isDistributionTab) return undefined;
    let maxCategories = 0;
    for (const panel of deferredPanels) {
      const keys = new Set<string>();
      for (const row of panel.rows) {
        keys.add(panel.reactantTypes.map((col) => String(row[col] ?? '(no value)')).join(' / '));
      }
      maxCategories = Math.max(maxCategories, keys.size);
    }
    return Math.max(800, maxCategories * 110);
  }, [isSplit, isDistributionTab, deferredPanels]);

  return (
    <div className="analysis-view">
      <div className="view-toggle-row">
        {isDistributionTab && (
          <button
            className={`eln-legend-btn${showElnLegend ? ' active' : ''}`}
            onClick={toggleElnLegend}
            title={showElnLegend ? 'Hide ELN count legend' : 'Show ELN count legend'}
          >
            Legend
          </button>
        )}
        <div className="view-toggle" id="view-toggle">
          {visibleTabs.map((tab) => (
            <button
              key={tab.id}
              className={`view-toggle-btn${effectiveTab === tab.id ? ' active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
              title={tab.label}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="view-content">
        {isSplit ? (
          <div className="split-grid">
            {deferredPanels.map((panel) => (
              <div key={panel.label} className="split-panel">
                <div className="split-panel-label">{panel.label}</div>
                <PanelWithComparison
                  tab={effectiveTab}
                  panel={panel}
                  comparisonRawData={comparisonRawData}
                  heightOverride={splitHeight}
                />
              </div>
            ))}
          </div>
        ) : (
          <div>
            <div className="split-panel-label">{reactantTypes.join(' / ')}</div>
            <PanelWithComparison
              tab={effectiveTab}
              panel={deferredPanels[0]}
              comparisonRawData={comparisonRawData}
            />
          </div>
        )}
      </div>
    </div>
  );
}
