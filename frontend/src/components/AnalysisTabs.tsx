import { useFilterStore } from '../stores/filterStore';
import { DistributionView } from './DistributionView';
import { createBoxplotConfig } from '../plots/boxplot';
import { createViolinConfig } from '../plots/violin';
import { HeatmapView } from './HeatmapView';
import { StatsTable } from './StatsTable';
import { useSplitFilteredData } from '../hooks/useSplitFilteredData';
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

function renderPanel(tab: TabId, panel: SplitPanel) {
  const { rows, reactantTypes } = panel;
  switch (tab) {
    case 'boxplot':
      return (
        <DistributionView
          buildConfig={createBoxplotConfig}
          label="boxplot"
          rows={rows}
          reactantTypes={reactantTypes}
        />
      );
    case 'violin':
      return (
        <DistributionView
          buildConfig={createViolinConfig}
          label="violin plot"
          rows={rows}
          reactantTypes={reactantTypes}
        />
      );
    case 'heatmap':
      return (
        <HeatmapView rows={rows} reactantTypes={reactantTypes} />
      );
    case 'stats':
      return (
        <StatsTable rows={rows} reactantTypes={reactantTypes} />
      );
  }
}

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
                {renderPanel(effectiveTab, panel)}
              </div>
            ))}
          </div>
        ) : (
          renderPanel(effectiveTab, panels[0])
        )}
      </div>
    </div>
  );
}
