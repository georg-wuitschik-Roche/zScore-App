import { memo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { DistributionView } from './DistributionView';
import { createBoxplotConfig } from '../plots/boxplot';
import { createViolinConfig } from '../plots/violin';
import { HeatmapView } from './HeatmapView';
import { StatsTable } from './StatsTable';
import { useSplitFilteredData } from '../hooks/useSplitFilteredData';
import { useComparisonRanks } from '../hooks/useComparisonData';
import type { ComparisonResult } from '../hooks/useComparisonData';
import type { TabId, SplitPanel } from '../data/types';

interface TabDef {
  id: TabId;
  label: string;
  requiresMultiReactant: boolean;
}

const TABS: TabDef[] = [
  { id: 'boxplot', label: 'Boxplot', requiresMultiReactant: false },
  { id: 'violin', label: 'Violin', requiresMultiReactant: false },
  { id: 'heatmap', label: 'Heatmap', requiresMultiReactant: true },
  { id: 'stats', label: 'Stats', requiresMultiReactant: false },
];

function renderPanel(tab: TabId, panel: SplitPanel, comparison: ComparisonResult | null) {
  const { rows, reactantTypes, stats } = panel;
  const noDataHint = stats.noDataHint;
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
        />
      );
    case 'heatmap':
      return (
        <HeatmapView
          rows={rows}
          reactantTypes={reactantTypes}
          noDataHint={noDataHint}
          axisRankMaps={comparison?.axisRankMaps}
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

/** Wrapper that calls useComparisonRanks per panel so each gets its own rank deltas. */
const PanelWithComparison = memo(function PanelWithComparison({ tab, panel }: { tab: TabId; panel: SplitPanel }) {
  const comparison = useComparisonRanks(panel.rows, panel.reactantTypes);
  return renderPanel(tab, panel, comparison);
});

export function AnalysisTabs() {
  const activeTab = useFilterStore((s) => s.activeTab);
  const setActiveTab = useFilterStore((s) => s.setActiveTab);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const splitSelector = useFilterStore((s) => s.splitSelector);
  const panels = useSplitFilteredData();

  const isSplit = panels.length > 1;

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

  return (
    <div className="analysis-view">
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

      <div className="view-content">
        {isSplit ? (
          <div className="split-grid">
            {panels.map((panel) => (
              <div key={panel.label} className="split-panel">
                <div className="split-panel-label">{panel.label}</div>
                <PanelWithComparison tab={effectiveTab} panel={panel} />
              </div>
            ))}
          </div>
        ) : (
          <PanelWithComparison tab={effectiveTab} panel={panels[0]} />
        )}
      </div>
    </div>
  );
}
